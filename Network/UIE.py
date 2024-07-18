'''
Author:Xuelei Chen(chenxuelei@hotmail.com)
'''
import torch
import numpy as np
import torch.nn as nn


class ANet(nn.Module):
    def __init__(self):
        super(ANet,self).__init__()

        block = [nn.Conv2d(1,3,3,padding = 1)]
        block += [nn.PReLU()]

        block += [nn.Conv2d(3,3,3,padding = 1)]
        block += [nn.PReLU()]

        block += [nn.AdaptiveAvgPool2d((1,1))]
        block += [nn.Conv2d(3,3,1)]
        block += [nn.PReLU()]
        block += [nn.Conv2d(3,1,1)]
        block += [nn.PReLU()]
        # block += [nn.ReLU()]
        self.block = nn.Sequential(*block)

    def forward(self,x):
        return self.block(x)

class TNet(nn.Module):
    def __init__(self):
        super(TNet,self).__init__()

        block = [nn.Conv2d(2,8,3,padding=1,dilation=1)]
        block += [nn.PReLU()]
        block += [nn.Conv2d(8,8,3,padding=2,dilation=2)]
        block += [nn.PReLU()]
        block += [nn.Conv2d(8,8,3,padding=5,dilation=5)]
        block += [nn.PReLU()]

        block += [nn.Conv2d(8,1,3,padding=1)]
        # block += [nn.PReLU()]
        block += [nn.Sigmoid()]
        # block += [nn.ReLU()]
        self.block = nn.Sequential(*block)
    def forward(self,x):
        return self.block(x)


# class ANet(nn.Module):
#     def __init__(self):
#         super(ANet,self).__init__()

#         block = [nn.Conv2d(3,3,3,padding = 1)]
#         block += [nn.PReLU()]

#         block += [nn.Conv2d(3,3,3,padding = 1)]
#         block += [nn.PReLU()]

#         block += [nn.AdaptiveAvgPool2d((1,1))]
#         block += [nn.Conv2d(3,3,1)]
#         block += [nn.PReLU()]
#         block += [nn.Conv2d(3,3,1)]
#         block += [nn.PReLU()]
#         self.block = nn.Sequential(*block)

#     def forward(self,x):
#         return self.block(x)

# class TNet(nn.Module):
#     def __init__(self):
#         super(TNet,self).__init__()

#         block = [nn.Conv2d(6,8,3,padding=1,dilation=1)]
#         block += [nn.PReLU()]
#         block += [nn.Conv2d(8,8,3,padding=2,dilation=2)]
#         block += [nn.PReLU()]
#         block += [nn.Conv2d(8,8,3,padding=5,dilation=5)]
#         block += [nn.PReLU()]

#         block += [nn.Conv2d(8,3,3,padding=1)]
#         block += [nn.PReLU()]
#         # block += [nn.Sigmoid()]
#         self.block = nn.Sequential(*block)
#     def forward(self,x):
#         return self.block(x)

        
class PhysicalNN(nn.Module):
    def __init__(self):
        super(PhysicalNN,self).__init__()

        self.ANet = ANet()
        self.tNet = TNet()

    def forward(self,x):
        A = self.ANet(x)
        t = self.tNet(torch.cat((x*0+A,x),1))
        out = ((x-A)*t + A)
        # breakpoint()
        return torch.clamp(out,0.,1.)
