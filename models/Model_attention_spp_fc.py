import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class Conv(nn.Module):
    def __init__(self,c_in,c_out,k=1,s=1,p=0):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c_in,c_out,k,s,p)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.Hardswish()
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class Focus(nn.Module): #
    def __init__(self,c_in=12,c_out=64): #c_in 为3*4，后面那个concat操作导致
        super(Focus, self).__init__()
        self.conv_f = Conv(c_in,c_out,3,1,1)
    def forward(self, x):  # x(b,c,h,w) -> y(b,4c,h/2,w/2) 切片后合并。
        return self.conv_f(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1))


class Bottleneck(nn.Module): #实际是借鉴resnet残差结构，类似的，注意中间的部分要保证特征图大小不变。 通道数量输入输出一致。
    def __init__(self,c_in,c_out,shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c_out * e)  # hidden channels
        self.bottleneck = nn.Sequential(
            Conv(c_in,c_), #此处，通道减少，同时融合通道。
            Conv(c_,c_out,3,1,1)
        )
        self.add = shortcut and c_in == c_out #如果输入输出通道不一致就不shortcut了
    def forward(self,x):
         return self.bottleneck(x)+x if self.add else self.bottleneck(x)
class BL(nn.Module): #concat之后的batchnormal+激活。
    def __init__(self,c_):
        super(BL, self).__init__()
        self.bl = nn.Sequential(
            nn.BatchNorm2d(c_),
            nn.LeakyReLU(0.1,True)
        )
    def forward(self,x):
        return self.bl(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class Bottleneckcsp(nn.Module): #c_ = c_out*e,e=0.5,默认缩放其通道的因子。
    def __init__(self,c_in,c_out,n=3,depth_multiple=0.33,e=0.5,shortcut=True): #n为重复bottleneck的数量。
        super(Bottleneckcsp, self).__init__()
        c_ = int(c_out*e)
        self.n = max(round(n*depth_multiple),1)
        self.conv1 = Conv(c_in,c_) #bottleneck之前的那个卷积。
        self.bottleneck = nn.ModuleList([Bottleneck(c_,c_,shortcut) for _ in range(self.n)])
        self.conv2 = nn.Conv2d(c_,c_,1,1, bias=False) #bottleneck之后的那个卷积 作用，融合通道。
        self.conv3 = nn.Conv2d(c_in,c_,1,1, bias=False) #跳跃连接的卷积，目的，调整通道和bottleneck的输出通道一致。也有融合通道的作用。
        #然后concat在forward中体现。
        self.bl = BL(c_*2)
        self.conv4 = Conv(c_*2,c_out,1,1) #确定输出通道数量。
        self.ca = ChannelAttention(c_out)  # 通道注意力
        self.sa = SpatialAttention()  # 空间注意力

    def forward(self,x):
        y = self.conv1(x)
        for i in range(self.n):
            y = self.bottleneck[i](y)
        y = self.conv2(y)
        y_shortcut = self.conv3(x)
        y = torch.cat((y,y_shortcut),dim=1) #通道上，拼接
        y = self.bl(y)
        y = self.conv4(y)
        y = self.ca(y) * y  # 广播机制
        y = self.sa(y) * y
        return y


class SPP(nn.Module):
    def __init__(self,c_in,c_out, k=(5, 9, 13)):
        super(SPP, self).__init__()
        self.conv1 = Conv(c_in,c_in//2,1,1)
        self.maxpool = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        '''[nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False), #padding实际上是卷积核的整除
            nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False),
            nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)]'''
        ##然后concat在forward中体现。
        self.conv2 = Conv(c_in*2,c_out,1,1)

    def forward(self,x):
        y = self.conv1(x)
        max_pool = [y]
        for m in self.maxpool:
            max_pool.append(m(y))
        y = torch.cat(max_pool,dim=1)
        y = self.conv2(y)
        return y

class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels=[3,2,1], pool_type='max_pool'):
        '''num_levels ,池化核n*n'''
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        # num:样本数量 c:通道数 h:高 w:宽
        # num: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        num, c, h, w = x.size()
        #         print(x.size())
        for i in range(len(self.num_levels)):
            level = self.num_levels[i]

            '''
            The equation is explained on the following site:
            http://www.cnblogs.com/marsggbo/p/8572846.html#autoid-0-0-0
            '''
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.floor(h / level), math.floor(w / level))
            pooling = (
            math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

            # update input data with padding
            zero_pad = torch.nn.ZeroPad2d((pooling[1], pooling[1], pooling[0], pooling[0]))
            x_new = zero_pad(x)

            # update kernel and stride
            h_new = 2 * pooling[0] + h
            w_new = 2 * pooling[1] + w

            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                try:
                    tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
                except Exception as e:
                    print(str(e))
                    print(x.size())
                    print(level)
            else:
                tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten

class Head(nn.Module):
    def __init__(self,c_init,nc):
        super(Head, self).__init__()
        self.conv = nn.Conv2d(c_init, 128, 1, 1) #n,128,h,w (13,14,15,16等)
        self.spplayer = SPPLayer(num_levels=[3,2,1]) #n,1792
        self.fc = nn.Sequential(nn.Linear(1792,256),
                                nn.Hardswish(inplace=True),
                                nn.Linear(256,nc))

        # self.act = nn.Sigmoid() #后面用逻辑斯蒂二回归
    def forward(self,x):
        x = self.conv(x)
        x = self.spplayer(x)
        x = self.fc(x)
        print(x.shape)
        return x


'''利用上面的模块写主网络'''
class MainNet(nn.Module):
    '''
    na是锚框anchor数量，nc是分类数量，depth_multiple是深度系数，width_multiple是宽度系数。
    yolov5s:
    depth_multiple: 0.33  # model depth multiple
    width_multiple: 0.50  # layer channel multiple
    yolov5m:
    depth_multiple: 0.67
    width_multiple: 0.75
    yolov5l:
    depth_multiple: 1.0
    width_multiple: 1.0
    yolov5x:
    depth_multiple: 1.33
    width_multiple: 1.25
    '''
    def __init__(self,nc=5,depth_multiple=0.33,width_multiple=0.5):#默认是yolov5s
        super(MainNet, self).__init__()
        c_init= math.ceil(64*width_multiple/8)*8
        # print('c_init',c_init) 32
        #backbone
        # self.focus = Focus(12,c_init) #12是3*4得来，具体看模块中的实现。
        # self.conv1 = Conv(c_init,c_init*2,3,2,1) #32,64
        # self.bottleneck_1_3 = Bottleneckcsp(c_init*2,c_init*2,n=3,depth_multiple=depth_multiple) #64,64
        # self.conv2 = Conv(c_init*2,c_init*4,3,2,1) #64,128
        # self.bottleneck_2_9 = Bottleneckcsp(c_init * 4, c_init * 4, n=9, depth_multiple=depth_multiple)  # 128,128
        # #此处输出给concat_fpn_2
        self.focus_2bottleneck = nn.Sequential(
            Focus(12, c_init),
            Conv(c_init, c_init * 2, 3, 2, 1),
            Bottleneckcsp(c_init * 2, c_init * 2, n=3, depth_multiple=depth_multiple),
            Conv(c_init * 2, c_init * 4, 3, 2, 1),
            Bottleneckcsp(c_init * 4, c_init * 4, n=9, depth_multiple=depth_multiple)
        )

        self.conv3 = Conv(c_init * 4, c_init * 8, 3, 2, 1)  # 128,256
        self.bottleneck_3_9 = Bottleneckcsp(c_init * 8, c_init * 8, n=9, depth_multiple=depth_multiple) #256,256
        # 此处输出给concat_fpn_1
        self.conv4 = Conv(c_init * 8, c_init * 16, 3, 2, 1)  # 256,512
        self.spp = SPP(c_init * 16,c_init * 16) #512,512
        self.bottleneck_4_3 = Bottleneckcsp(c_init * 16, c_init * 16, n=3, depth_multiple=depth_multiple) #512,512
        #输出给 conv_fpn_1

        #neck FPN
        self.conv_fpn_1 = Conv(c_init * 16,c_init * 8,1,1) #512,256
        # 此处输出给 conv_pan_2合并
        #此处上采样在forward中体现
        #此处concat_fpn_1 #512
        self.bottleneck_fpn_1_3 = Bottleneckcsp(c_init * 16, c_init * 8, n=3, depth_multiple=depth_multiple) #512,256
        self.conv_fpn_2 = Conv(c_init * 8,c_init * 4,1,1)#256,128
        #此处输出给 conv_pan_1合并
        #此处上采样
        #此处concat_fpn_2 #256
        #此处concat_fpn_2合并后，输出给 bottleneck_pan_1_3

        #neck pan
        self.bottleneck_pan_1_3 = Bottleneckcsp(c_init * 8, c_init * 4, n=3, depth_multiple=depth_multiple) #256,128
        #输出给 head_1
        self.conv_pan_1 = Conv(c_init * 4,c_init * 4,3,2,1) #128,128
        #此处concat_pan_1 #256
        self.bottleneck_pan_2_3 = Bottleneckcsp(c_init * 8, c_init * 8, n=3, depth_multiple=depth_multiple) #256,256
        #此处输出给head_2
        self.conv_pan_2 = Conv(c_init * 8,c_init * 8,3,2,1) #256,256
        #此处concat_pan_2 #512
        self.bottleneck_pan_3_3 = Bottleneckcsp(c_init * 16, c_init * 16, n=3, depth_multiple=depth_multiple) #512,512
        #此处输出给head_3

        #head
        self.head_1 = Head(c_init * 4,nc)
        self.head_2 = Head(c_init * 8, nc)
        self.head_3 = Head(c_init * 16, nc)
    def forward(self,x):
        #backbone
        focus_2bottleneck = self.focus_2bottleneck(x) #此处输出给concat_fpn_2
        conv3 = self.conv3(focus_2bottleneck)
        bottleneck_3_9 = self.bottleneck_3_9(conv3) #此处输出给concat_fpn_1
        conv4 = self.conv4(bottleneck_3_9)
        spp = self.spp(conv4)
        bottleneck_4_3 = self.bottleneck_4_3(spp) #输出给 conv_fpn_1
        #neck fpn
        conv_fpn_1 = self.conv_fpn_1(bottleneck_4_3) #此处输出给 conv_pan_2合并
        up_fpn_1 = F.interpolate(conv_fpn_1, scale_factor=2, mode='nearest')
        concat_fpn_1 = torch.cat((up_fpn_1,bottleneck_3_9),dim=1)
        bottleneck_fpn_1_3 = self.bottleneck_fpn_1_3(concat_fpn_1)
        conv_fpn_2 = self.conv_fpn_2(bottleneck_fpn_1_3) #此处输出给 conv_pan_1合并
        up_fpn_2 = F.interpolate(conv_fpn_2, scale_factor=2, mode='nearest')
        concat_fpn_2 = torch.cat((up_fpn_2,focus_2bottleneck),dim=1) #输出给bottleneck_pan_1_3
        #neck pan
        bottleneck_pan_1_3 = self.bottleneck_pan_1_3(concat_fpn_2) #输出给head_1
        conv_pan_1 = self.conv_pan_1(bottleneck_pan_1_3)
        concat_pan_1 = torch.cat((conv_pan_1,conv_fpn_2),dim=1)
        bottleneck_pan_2_3 = self.bottleneck_pan_2_3(concat_pan_1) #输出给head_2
        conv_pan_2 = self.conv_pan_2(bottleneck_pan_2_3)
        concat_pan_2 = torch.cat((conv_pan_2,conv_fpn_1),dim=1)
        bottleneck_pan_3_3 = self.bottleneck_pan_3_3(concat_pan_2) #输出给head_3
        #head
        head_1 = self.head_1(bottleneck_pan_1_3)
        head_2 = self.head_2(bottleneck_pan_2_3)
        head_3 = self.head_3(bottleneck_pan_3_3)

        return head_1,head_2,head_3


if __name__ == '__main__':
    #416,448,480,512
    input = torch.ones(2,3,480,480)
    #416: 52,26,13
    #448:56,28,14
    #480:60,30,15
    #512:64,32,16
    # print(input.shape
    net = MainNet(depth_multiple=1,width_multiple=1)
    # net = MainNet()
    head_1,head_2,head_3 = net(input)
    print(head_1,head_2.shape,head_3.shape)
    # print(net)
