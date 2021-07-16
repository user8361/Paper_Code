
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)





class demo(nn.Module):
    def __init__(self):
        # 增加 Coordinate Attention 部分 ------------------- #

        self.ca_squeeze =  nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.ca_expand =  nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.ca_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.ca_pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.ca_bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.ca_act = h_swish()

        self.ca_conv_h = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=1,
                                   stride=1, padding=0)
        self.ca_conv_w = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=1,
                                   stride=1, padding=0)

        # -------------------------------------------------- #

    def forward(self,x):

        # only coordinate attention
        x_ca_squeeze = self.ca_squeeze(x_shift)
        x_ca_h = self.ca_pool_h(x_ca_squeeze)
        x_ca_w = self.ca_pool_w(x_ca_squeeze).permute(0,1,3,2)

        x_ca = torch.cat([x_ca_h,x_ca_w],dim = 2)
        x_ca = self.ca_bn1(x_ca)
        x_ca = self.ca_act(x_ca)

        x_ca_h,x_ca_w =torch.split(x_ca,[h,w],dim=2)
        x_ca_w = x_ca_w.permute(0,1,3,2)

        x_ca_h_conv = self.ca_conv_h(x_ca_h)
        x_ca_h_expand = self.ca_expand(x_ca_h_conv)
        x_ca_h_attention = self.sigmoid(x_ca_h_expand)

        x_ca_w_conv = self.ca_conv_w(x_ca_w)
        x_ca_w_expand = self.ca_expand(x_ca_w_conv)
        x_ca_w_attention = self.sigmoid(x_ca_w_expand)

        x_ca_out = x_shift * x_ca_h_attention*x_ca_w_attention
        
        
        
