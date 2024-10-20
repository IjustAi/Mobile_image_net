import torch 
import torch.nn as nn 
import math 

__all__ = ['mobilenetv2'] 

def _make_divisible(v,divisor,min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value,int(v+divisor/2)//divisor*divisor)
    if new_v <0.9*v:
        new_v +=divisor
    return new_v

def conv_3(inp,out,stride):
    return nn.Sequential(
        nn.Conv2d(inp,out,3,stride,1,bias=False),
        nn.BatchNorm2d(out),
        nn.ReLU6(inplace=True)
    )

def conv_1(inp,out):
    return nn.Sequential(
        nn.Conv2d(inp,out,1,1,0,bias=False),
        nn.BatchNorm2d(out),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self,inp,out,stride,expand_ratio):
        super(InvertedResidual,self).__init__()
        assert stride in [1,2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride==1 and inp ==out

        if expand_ratio ==1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3,stride,1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #pw-linear
                nn.Conv2d(hidden_dim,out,1,1,0,bias=False),
                nn.BatchNorm2d(out)
            )

        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim,1,1,0,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                nn.Conv2d(hidden_dim, hidden_dim,3,stride,1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                nn.Conv2d(hidden_dim,out,1,1,0,bias=False),
                nn.BatchNorm2d(out)
            )
    def forward(self,x):
        if self.identity:
            return x+self.conv(x)
        else:
            return self.conv(x)

class MobileNetv2(nn.Module):
    def __init__(self,num_classes=100,width_mult=1.):
        super(MobileNetv2,self).__init__()

        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3(3, input_channel, 2)]
        block = InvertedResidual

        for t,c,n,s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel

        self.features = nn.Sequential(*layers)
        output_channel = output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1(input_channel,output_channel)
        self.avgpool =nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(output_channel,num_classes)
        self._initialize_weights()

    def forward(self,x):
        x= self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x= self.classifier(x)
        return x 

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] *m.kernel_size[1] *m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    return MobileNetv2(**kwargs)
