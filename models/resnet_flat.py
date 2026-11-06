import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
        self.bath1 = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        return self.bath1(self.conv1(x))
    
class ResNet20_Flat(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20_Flat, self).__init__()
        self.conv_initial = ConvBlock(3,16,3,1,1)
        
        self.conv_block111 = ConvBlock(16,16,3,1,1)
        self.conv_block112 = ConvBlock(16,16,3,1,1)
        self.conv_block121 = ConvBlock(16,16,3,1,1)
        self.conv_block122 = ConvBlock(16,16,3,1,1)
        self.conv_block131 = ConvBlock(16,16,3,1,1)
        self.conv_block132 = ConvBlock(16,16,3,1,1)
        
        self.conv_block211 = ConvBlock(16,32,3,1,2)
        self.conv_block212 = ConvBlock(32,32,3,1,1)
        self.conv_block210 = ConvBlock(16,32,1,0,2)
        self.conv_block221 = ConvBlock(32,32,3,1,1)
        self.conv_block222 = ConvBlock(32,32,3,1,1)
        self.conv_block231 = ConvBlock(32,32,3,1,1)
        self.conv_block232 = ConvBlock(32,32,3,1,1)
        
        self.conv_block311 = ConvBlock(32,64,3,1,2)
        self.conv_block312 = ConvBlock(64,64,3,1,1)
        self.conv_block310 = ConvBlock(32,64,1,0,2)
        self.conv_block321 = ConvBlock(64,64,3,1,1)
        self.conv_block322 = ConvBlock(64,64,3,1,1)
        self.conv_block331 = ConvBlock(64,64,3,1,1)
        self.conv_block332 = ConvBlock(64,64,3,1,1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        
    def forward(self, x):
        x = F.relu(self.conv_initial(x))
        
        x11 = F.relu(self.conv_block111(x))
        x = x + self.conv_block112(x11)
        x = F.relu(x)
        x12 = F.relu(self.conv_block121(x))
        x = x + self.conv_block122(x12)
        x = F.relu(x)
        x13 = F.relu(self.conv_block131(x))
        x = x + self.conv_block132(x13)
        x = F.relu(x)
        
        x21 = F.relu(self.conv_block211(x))
        x20 = self.conv_block210(x)
        x = self.conv_block212(x21) + x20
        x = F.relu(x)
        x22 = F.relu(self.conv_block221(x))
        x = x + self.conv_block222(x22)
        x = F.relu(x)
        x23 = F.relu(self.conv_block231(x))
        x = x + self.conv_block232(x23)
        x = F.relu(x)
        
        x31 = F.relu(self.conv_block311(x))
        x30 = self.conv_block310(x)
        x = self.conv_block312(x31) + x30
        x = F.relu(x)
        x32 = F.relu(self.conv_block321(x))
        x = x + self.conv_block322(x32)
        x = F.relu(x)
        x33 = F.relu(self.conv_block331(x))
        x = x + self.conv_block332(x33)
        x = F.relu(x)
        
        out = self.avgpool(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out
        
    
    def inputs_ext(self, x):
        out11 = F.relu(self.conv_initial(x))
        
        out112 = F.relu(self.conv_block111(out11))
        out12 = out11 + self.conv_block112(out112)
        out12 = F.relu(out12)
        out121 = F.relu(self.conv_block121(out12))
        out13 = out12 + self.conv_block122(out121)
        out13 = F.relu(out13)
        out131 = F.relu(self.conv_block131(out13))
        out14 = out13 + self.conv_block132(out131)
        out14 = F.relu(out14)
        
        out211 = F.relu(self.conv_block211(out14))
        out20 = self.conv_block210(out14)
        out21 = self.conv_block212(out211) + out20
        out21 = F.relu(out21)
        out221 = F.relu(self.conv_block221(out21))
        out22 = out21 + self.conv_block222(out221)
        out22 = F.relu(out22)
        out231 = F.relu(self.conv_block231(out22))
        out23 = out22 + self.conv_block232(out231)
        out23 = F.relu(out23)
        
        out311 = F.relu(self.conv_block311(out23))
        out30 = self.conv_block310(out23)
        out31 = self.conv_block312(out311) + out30
        out31 = F.relu(out31)
        out321 = F.relu(self.conv_block321(out31))
        out32 = out31 + self.conv_block322(out321)
        out32 = F.relu(out32)
        out331 = F.relu(self.conv_block331(out32))
        out33 = out32 + self.conv_block332(out331)
        out33 = F.relu(out33)
        
        out33 = self.avgpool(out33)
        out33 = torch.flatten(out33, 1)
        out = self.fc(out33)
        
        return [x, out11, out112, out12, out121, out13, out131, 
                out14, out221, out14, out21, out221, out22, out231, 
                out23, out311, out23, out31, out321, out32, out331, out33, out]

class Wide_ResNet18_Flat(nn.Module):
    def __init__(self, num_classes=10):
        super(Wide_ResNet18_Flat, self).__init__()
        self.conv_initial = ConvBlock(3,64,3,1,1)
        
        self.conv_block111 = ConvBlock(64,64,3,1,1)
        self.conv_block112 = ConvBlock(64,64,3,1,1)
        self.conv_block121 = ConvBlock(64,64,3,1,1)
        self.conv_block122 = ConvBlock(64,64,3,1,1)
        
        self.conv_block211 = ConvBlock(64,128,3,1,2)
        self.conv_block212 = ConvBlock(128,128,3,1,1)
        self.conv_block210 = ConvBlock(64,128,1,0,2)
        self.conv_block221 = ConvBlock(128,128,3,1,1)
        self.conv_block222 = ConvBlock(128,128,3,1,1)
        
        self.conv_block311 = ConvBlock(128,256,3,1,2)
        self.conv_block312 = ConvBlock(256,256,3,1,1)
        self.conv_block310 = ConvBlock(128,256,1,0,2)
        self.conv_block321 = ConvBlock(256,256,3,1,1)
        self.conv_block322 = ConvBlock(256,256,3,1,1)
        
        self.conv_block411 = ConvBlock(256,512,3,1,2)
        self.conv_block412 = ConvBlock(512,512,3,1,1)
        self.conv_block410 = ConvBlock(256,512,1,0,2)
        self.conv_block421 = ConvBlock(512,512,3,1,1)
        self.conv_block422 = ConvBlock(512,512,3,1,1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        
    def forward(self, x):
        x = F.relu(self.conv_initial(x))
        
        x11 = F.relu(self.conv_block111(x))
        x = x + self.conv_block112(x11)
        x = F.relu(x)
        x12 = F.relu(self.conv_block121(x))
        x = x + self.conv_block122(x12)
        x = F.relu(x)
        
        x21 = F.relu(self.conv_block211(x))
        x20 = self.conv_block210(x)
        x = self.conv_block212(x21) + x20
        x = F.relu(x)
        x22 = F.relu(self.conv_block221(x))
        x = x + self.conv_block222(x22)
        x = F.relu(x)
        
        x31 = F.relu(self.conv_block311(x))
        x30 = self.conv_block310(x)
        x = self.conv_block312(x31) + x30
        x = F.relu(x)
        x32 = F.relu(self.conv_block321(x))
        x = x + self.conv_block322(x32)
        x = F.relu(x)
        
        x41 = F.relu(self.conv_block411(x))
        x40 = self.conv_block410(x)
        x = self.conv_block412(x41) + x40
        x = F.relu(x)
        x42 = F.relu(self.conv_block421(x))
        x = x + self.conv_block422(x42)
        x = F.relu(x)
        
        out = self.avgpool(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out
        
    
    def inputs_ext(self, x):
        out11 = F.relu(self.conv_initial(x))
        
        out112 = F.relu(self.conv_block111(out11))
        out12 = out11 + self.conv_block112(out112)
        out12 = F.relu(out12)
        out121 = F.relu(self.conv_block121(out12))
        out13 = out12 + self.conv_block122(out121)
        out13 = F.relu(out13)
        
        out211 = F.relu(self.conv_block211(out13))
        out20 = self.conv_block210(out13)
        out21 = self.conv_block212(out211) + out20
        out21 = F.relu(out21)
        out221 = F.relu(self.conv_block221(out21))
        out22 = out21 + self.conv_block222(out221)
        out22 = F.relu(out22)
        
        out311 = F.relu(self.conv_block311(out22))
        out30 = self.conv_block310(out22)
        out31 = self.conv_block312(out311) + out30
        out31 = F.relu(out31)
        out321 = F.relu(self.conv_block321(out31))
        out32 = out31 + self.conv_block322(out321)
        out32 = F.relu(out32)
        
        out411 = F.relu(self.conv_block411(out32))
        out40 = self.conv_block410(out32)
        out41 = self.conv_block412(out411) + out40
        out41 = F.relu(out41)
        out421 = F.relu(self.conv_block421(out41))
        out42 = out41 + self.conv_block422(out421)
        out42 = F.relu(out42)
        
        out42 = self.avgpool(out42)
        out42 = torch.flatten(out42, 1)
        out = self.fc(out42)
        
        return [x, out11, out112, out12, out121, 
                out13, out221, out13, out21, out221,  
                out22, out311, out22, out31, out321,
                out32, out411, out32, out41, out421, out42, out]

class ConvBlock_Relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride, is_relu=True):
        super(ConvBlock_Relu, self).__init__()
        self.is_relu = is_relu
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
        self.bath1 = nn.BatchNorm2d(out_channel)
        if is_relu:
            self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        if self.is_relu:
            return self.relu(self.bath1(self.conv1(x)))
        return self.bath1(self.conv1(x))

class ResNet50_Flat(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet50_Flat, self).__init__()
        self.conv_initial = ConvBlock_Relu(3,64,3,1,1)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_block110 = ConvBlock_Relu(64,256,1,0,1, False)
        self.conv_block111 = ConvBlock_Relu(64,64,1,0,1)
        self.conv_block112 = ConvBlock_Relu(64,64,3,1,1)
        self.conv_block113 = ConvBlock_Relu(64,256,1,0,1, False)
        self.conv_block121 = ConvBlock_Relu(256,64,1,0,1)
        self.conv_block122 = ConvBlock_Relu(64,64,3,1,1)
        self.conv_block123 = ConvBlock_Relu(64,256,1,0,1, False)
        self.conv_block131 = ConvBlock_Relu(256,64,1,0,1)
        self.conv_block132 = ConvBlock_Relu(64,64,3,1,1)
        self.conv_block133 = ConvBlock_Relu(64,256,1,0,1, False)
        
        self.conv_block210 = ConvBlock_Relu(256,512,1,0,2, False)
        self.conv_block211 = ConvBlock_Relu(256,128,1,0,1)
        self.conv_block212 = ConvBlock_Relu(128,128,3,1,2)
        self.conv_block213 = ConvBlock_Relu(128,512,1,0,1, False)
        self.conv_block221 = ConvBlock_Relu(512,128,1,0,1)
        self.conv_block222 = ConvBlock_Relu(128,128,3,1,1)
        self.conv_block223 = ConvBlock_Relu(128,512,1,0,1, False)
        self.conv_block231 = ConvBlock_Relu(512,128,1,0,1)
        self.conv_block232 = ConvBlock_Relu(128,128,3,1,1)
        self.conv_block233 = ConvBlock_Relu(128,512,1,0,1, False)
        self.conv_block241 = ConvBlock_Relu(512,128,1,0,1)
        self.conv_block242 = ConvBlock_Relu(128,128,3,1,1)
        self.conv_block243 = ConvBlock_Relu(128,512,1,0,1, False)
        
        self.conv_block310 = ConvBlock_Relu(512,1024,1,0,2, False)
        self.conv_block311 = ConvBlock_Relu(512,256,1,0,1)
        self.conv_block312 = ConvBlock_Relu(256,256,3,1,2)
        self.conv_block313 = ConvBlock_Relu(256,1024,1,0,1, False)
        self.conv_block321 = ConvBlock_Relu(1024,256,1,0,1)
        self.conv_block322 = ConvBlock_Relu(256,256,3,1,1)
        self.conv_block323 = ConvBlock_Relu(256,1024,1,0,1, False)
        self.conv_block331 = ConvBlock_Relu(1024,256,1,0,1)
        self.conv_block332 = ConvBlock_Relu(256,256,3,1,1)
        self.conv_block333 = ConvBlock_Relu(256,1024,1,0,1, False)
        self.conv_block341 = ConvBlock_Relu(1024,256,1,0,1)
        self.conv_block342 = ConvBlock_Relu(256,256,3,1,1)
        self.conv_block343 = ConvBlock_Relu(256,1024,1,0,1, False)
        self.conv_block351 = ConvBlock_Relu(1024,256,1,0,1)
        self.conv_block352 = ConvBlock_Relu(256,256,3,1,1)
        self.conv_block353 = ConvBlock_Relu(256,1024,1,0,1, False)
        self.conv_block361 = ConvBlock_Relu(1024,256,1,0,1)
        self.conv_block362 = ConvBlock_Relu(256,256,3,1,1)
        self.conv_block363 = ConvBlock_Relu(256,1024,1,0,1, False)
        
        self.conv_block410 = ConvBlock_Relu(1024,2048,1,0,2, False)
        self.conv_block411 = ConvBlock_Relu(1024,512,1,0,1)
        self.conv_block412 = ConvBlock_Relu(512,512,3,1,2)
        self.conv_block413 = ConvBlock_Relu(512,2048,1,0,1, False)
        self.conv_block421 = ConvBlock_Relu(2048,512,1,0,1)
        self.conv_block422 = ConvBlock_Relu(512,512,3,1,1)
        self.conv_block423 = ConvBlock_Relu(512,2048,1,0,1, False)
        self.conv_block431 = ConvBlock_Relu(2048,512,1,0,1)
        self.conv_block432 = ConvBlock_Relu(512,512,3,1,1)
        self.conv_block433 = ConvBlock_Relu(512,2048,1,0,1, False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
        
    def forward(self, x):
        x = self.conv_initial(x)
        
        x110 = self.conv_block110(x)
        x111 = self.conv_block111(x)
        x112 = self.conv_block112(x111)
        x    = x110 + self.conv_block113(x112)
        x    = self.relu(x)
        x121 = self.conv_block121(x)
        x122 = self.conv_block122(x121)
        x    = x + self.conv_block123(x122)
        x    = self.relu(x)
        x131 = self.conv_block131(x)
        x132 = self.conv_block132(x131)
        x    = x + self.conv_block133(x132)
        x    = self.relu(x)
        
        x210 = self.conv_block210(x)
        x211 = self.conv_block211(x)
        x212 = self.conv_block212(x211)
        x    = x210 + self.conv_block213(x212)
        x    = self.relu(x)
        x221 = self.conv_block221(x)
        x222 = self.conv_block222(x221)
        x    = x + self.conv_block223(x222)
        x    = self.relu(x)
        x231 = self.conv_block231(x)
        x232 = self.conv_block232(x231)
        x    = x + self.conv_block233(x232)
        x    = self.relu(x)
        x241 = self.conv_block241(x)
        x242 = self.conv_block242(x241)
        x    = x + self.conv_block243(x242)
        x    = self.relu(x)
        
        x310 = self.conv_block310(x)
        x311 = self.conv_block311(x)
        x312 = self.conv_block312(x311)
        x    = x310 + self.conv_block313(x312)
        x    = self.relu(x)
        x321 = self.conv_block321(x)
        x322 = self.conv_block322(x321)
        x    = x + self.conv_block323(x322)
        x    = self.relu(x)
        x331 = self.conv_block331(x)
        x332 = self.conv_block332(x331)
        x    = x + self.conv_block333(x332)
        x    = self.relu(x)
        x341 = self.conv_block341(x)
        x342 = self.conv_block342(x341)
        x    = x + self.conv_block343(x342)
        x    = self.relu(x)
        x351 = self.conv_block351(x)
        x352 = self.conv_block352(x351)
        x    = x + self.conv_block353(x352)
        x    = self.relu(x)
        x361 = self.conv_block361(x)
        x362 = self.conv_block362(x361)
        x    = x + self.conv_block363(x362)
        x    = self.relu(x)
        
        x410 = self.conv_block410(x)
        x411 = self.conv_block411(x)
        x412 = self.conv_block412(x411)
        x    = x410 + self.conv_block413(x412)
        x    = self.relu(x)
        x421 = self.conv_block421(x)
        x422 = self.conv_block422(x421)
        x    = x + self.conv_block423(x422)
        x    = self.relu(x)
        x431 = self.conv_block431(x)
        x432 = self.conv_block432(x431)
        x    = x + self.conv_block433(x432)
        x    = self.relu(x)
        
        out = self.avgpool(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out
        
    
    def inputs_ext(self, x):
        
        out0 = self.conv_initial(x)
        
        out110 = self.conv_block110(out0)
        out111 = self.conv_block111(out0)
        out112 = self.conv_block112(out111)
        out1   = out110 + self.conv_block113(out112)
        out1   = self.relu(out1)
        out121 = self.conv_block121(out1)
        out122 = self.conv_block122(out121)
        out2   = out1 + self.conv_block123(out122)
        out2   = self.relu(out2)
        out131 = self.conv_block131(out2)
        out132 = self.conv_block132(out131)
        out3   = out2 + self.conv_block133(out132)
        out3   = self.relu(out3)
        
        out210 = self.conv_block210(out3)
        out211 = self.conv_block211(out3)
        out212 = self.conv_block212(out211)
        out4   = out210 + self.conv_block213(out212)
        out4   = self.relu(out4)
        out221 = self.conv_block221(out4)
        out222 = self.conv_block222(out221)
        out5    = out4 + self.conv_block223(out222)
        out5    = self.relu(out5)
        out231 = self.conv_block231(out5)
        out232 = self.conv_block232(out231)
        out6    = out5 + self.conv_block233(out232)
        out6    = self.relu(out6)
        out241 = self.conv_block241(out6)
        out242 = self.conv_block242(out241)
        out7    = out6 + self.conv_block243(out242)
        out7    = self.relu(out7)
        
        x310 = self.conv_block310(out7)
        x311 = self.conv_block311(out7)
        x312 = self.conv_block312(x311)
        out8    = x310 + self.conv_block313(x312)
        out8    = self.relu(out8)
        x321 = self.conv_block321(out8)
        x322 = self.conv_block322(x321)
        out9    = out8 + self.conv_block323(x322)
        out9    = self.relu(out9)
        x331 = self.conv_block331(out9)
        x332 = self.conv_block332(x331)
        out10    = out9 + self.conv_block333(x332)
        out10    = self.relu(out10)
        x341 = self.conv_block341(out10)
        x342 = self.conv_block342(x341)
        out11    = out10 + self.conv_block343(x342)
        out11    = self.relu(out11)
        x351 = self.conv_block351(out11)
        x352 = self.conv_block352(x351)
        out12    = out11 + self.conv_block353(x352)
        out12    = self.relu(out12)
        x361 = self.conv_block361(out12)
        x362 = self.conv_block362(x361)
        out13    = out12 + self.conv_block363(x362)
        out13    = self.relu(out13)
        
        x410 = self.conv_block410(out13)
        x411 = self.conv_block411(out13)
        x412 = self.conv_block412(x411)
        out14    = x410 + self.conv_block413(x412)
        out14    = self.relu(out14)
        x421 = self.conv_block421(out14)
        x422 = self.conv_block422(x421)
        out15    = out14 + self.conv_block423(x422)
        out15    = self.relu(out15)
        x431 = self.conv_block431(out15)
        x432 = self.conv_block432(x431)
        out16    = out15 + self.conv_block433(x432)
        out16    = self.relu(out16)
        
        #print(out16.size())
        out = self.avgpool(out16)
        #print(out.size())
        out = torch.flatten(out, 1)
        #print(out.size())
        f_out = self.fc(out)
        
        return [x, out0, out0, out111, out112, out1, out121, out122, out2, out131, out132, 
                out3, out3, out211, out212, out4, out221, out222, out5, out231, out232, out6, out241, out242, 
                out7, out7, x311, x312, out8, x321, x322, out9, x331, x332, out10, x341, x342, out11, x351, x352, out12, x361, x362,
                out13, out13, x411, x412, out14, x421, x422, out15, x431, x432, out, f_out]
