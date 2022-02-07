from xml.sax.xmlreader import InputSource
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
import numpy as np
import math


def TernaryTanh(x):
    st = time.time()
    x = torch.tanh(x)
    f1 = (x > 0.66).float()
    f2 = (x > 0.33).float()
    x = (f1 + f2) / 2
    return x

def Softmax(x):
    print('softmax')
    return nn.softmax(x)

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

class Dense(nn.Module):
    def __init__(self, in_channels, out_channels, before = None, after = False, bias=True, device=None, dtype=None):
        super(Dense, self).__init__()
        self.dense = nn.Linear(in_channels, out_channels, bias = bias)
        self.dense.apply(weights_init('gaussian'))

        if after=='BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after=='Tanh':
            self.after = torch.tanh
        elif after=='sigmoid':
            self.after = torch.sigmoid
        elif after == 'ternaryTanh':
            self.after = TernaryTanh
        elif after == 'softmax':
            self.after = nn.Softmax(dim = 1)

        if before=='ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before=='LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.dense(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x

class Cvi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=True):
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv.apply(weights_init('orthogonal'))

        if after=='BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after=='Tanh':
            self.after = torch.tanh
        elif after=='sigmoid':
            self.after = torch.sigmoid
        elif after == 'ternaryTanh':
            self.after = TernaryTanh
        elif after == 'softmax':
            self.after = nn.Softmax(dim = 1)
        elif(after == 'maxPooling'):
            self.after = nn.MaxPool2d(kernel_size = 2)
        elif(after == 'flatten'):
            self.after = nn.Flatten

        if before=='ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before=='LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x


class CvTi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=True):
        super(CvTi, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = bias)
        self.conv.apply(weights_init('orthogonal'))

        if after=='BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after=='Tanh':
            self.after = torch.tanh
        elif after=='sigmoid':
            self.after = torch.sigmoid
        elif after == 'TernaryTanh':
            self.after = TernaryTanh
        elif after == 'softmax':
            self.after = Softmax
        elif(after == 'maxPooling'):
            self.after = nn.MaxPool2d()

        if before=='ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before=='LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x

class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(Generator, self).__init__()

        self.Cv0 = Cvi(input_channels, 16, padding = 0, after = 'BN')

        self.Cv1 = Cvi(16, 32, before='LReLU', after='BN', stride = 1, padding = 5, dilation = 3)

        self.Cv2 = Cvi(32, 64, before='LReLU', after='BN')

        self.Cv3 = Cvi(64, 64, before='LReLU', after='BN', stride = 1)

        self.Cv4 = Cvi(64, 128, before='LReLU', after='BN', stride = 1)

        self.CvT5 = CvTi(128, 64, before='ReLU', after='BN', stride = 1)

        self.CvT6 = CvTi(128, 64, before='ReLU', after='BN', stride = 1)

        self.CvT7 = CvTi(128, 32, before='ReLU', after='BN', padding = 1, dilation = 1)

        self.CvT8 = CvTi(64, 16, before='ReLU', after='BN', stride = 1, padding = 2, dilation = 2)

        self.CvT9 = CvTi(32, output_channels, before='ReLU', after='Tanh', padding = 0)

    def forward(self, input):
        #encoder
        x0 = self.Cv0(input)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        x4 = self.Cv4(x3)
        x5 = self.CvT5(x4)

        cat1 = torch.cat([x5, x3], dim=1)
        x6 = self.CvT6(cat1)

        cat2 = torch.cat([x6, x2], dim=1)
        x7 = self.CvT7(cat2)

        cat3 = torch.cat([x7, x1], dim=1)
        x8 = self.CvT8(cat3)

        cat4 = torch.cat([x8, x0], dim=1)
        out = self.CvT9(cat4)

        out = (out + 1)/2



        return out

class Discriminator(nn.Module):
    def __init__(self, input_channels=4):
        super(Discriminator, self).__init__()

        self.Cv0 = Cvi(input_channels, 16)

        self.Cv1 = Cvi(16, 32, before='LReLU', after='BN', kernel_size = 3)

        self.Cv2 = Cvi(32, 64, before='LReLU', after='BN', kernel_size = 3)

        self.Cv3 = Cvi(64, 128, before='LReLU', after='BN', kernel_size = 3)

        self.Cv4 = Cvi(128, 128, before='LReLU', after='softmax', kernel_size = 3)

        self.l1 = Dense(1536, 512, before = 'ReLu' , after = 'softmax')

        self.l2 = Dense(512, 128, before = 'ReLu' , after = 'softmax')

        self.l3 = Dense(128, 2, before = 'ReLu' , after = 'softmax')

        self.Cv8 = Cvi(512, 2, before='LReLU', after='softmax', kernel_size = 3)

    def forward(self, input):
        x0 = self.Cv0(input)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        x4 = self.Cv4(x3)
        x4 = x4.view(x4.size(0),-1)
        x5 = self.l1(x4)
        x6 = self.l2(x5)
        out = self.l3(x6)

        return out

if __name__ == '__main__':
    #BCHW
    '''size = (8, 1, 88, 128)
    input = torch.ones(size)
    l1 = nn.L1Loss()
    input.requires_grad = True

    #convolution test
    conv = Cvi(3, 3)
    conv2 = Cvi(3, 3, before='ReLU', after='BN')
    output = conv(input)
    output2 = conv2(output)
    print(output.shape)
    print(output2.shape)
    loss = l1(output, torch.randn(3, 3, 128, 128))
    loss.backward()
    print(loss.item())

    convT = CvTi(3, 3)
    outputT = convT(output)
    print(outputT.shape)'''


    #Generator test
    print("GENERATOR")
    model = Generator(input_channels=1)
    output = model(input)
    print(output.shape)
    print(output)
    loss = l1(output, torch.randn(8, 1, 88, 128))
    loss.backward()
    print(loss.item())

    #Discriminator test
    size = (3, 4, 256, 256)
    input = torch.ones(size)
    l1 = nn.L1Loss()
    input.requires_grad = True
    model = Discriminator()
    output = model(input)
    print(output.shape)
