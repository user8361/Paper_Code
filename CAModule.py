import torch
import torch.nn as nn

class CAModule(nn.Module):
    def __init__(self, channel,nt, h, w, reduction=16):
        super(CAModule, self).__init__()

        self.nt = nt
        self.channel = channel
        self.reduction = reduction



        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((self.h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, self.w))

        self.conv_1x1 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel // self.reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.channel // self.reduction)

        self.F_h = nn.Conv2d(in_channels=self.channel // self.reduction, out_channels=self.channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=self.channel // self.reduction, out_channels=self.channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        residual = x
        print('CAModule ==>  ca input.shape ',x.shape)
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2) # 将 h w -> w h
        x_w = self.avg_pool_y(x) # h w

        print('CAModule ==>  x_h.shape ', x_h.shape) #8 48 1 224
        print('CAModule ==>  x_w.shape ', x_w.shape) #8 48 1 224

        x_cat_h_w =torch.cat((x_h,x_w),3)
        print('CAModule ==>  x_cat_h_w.shape ', x_cat_h_w.shape)
        x_cat_conv = self.conv_1x1(x_cat_h_w)
        print('CAModule ==>  x_cat_conv.shape ', x_cat_conv.shape)

        x_cat_conv_relu = self.relu(x_cat_conv)

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
        print('CAModule ==>  x_cat_conv_split_h.shape ', x_cat_conv_split_h.shape)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        print('CAModule ==>  s_h.shape ', s_h.shape)
        print('CAModule ==>  s_w.shape ', s_w.shape)

        out = x * s_h.expand_as(x) * s_w.expand_as(x) +residual
        print('CAModule ==>  out.shape ', out.shape)

        return out
