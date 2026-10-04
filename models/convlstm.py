import torch
import torch.nn as nn
from .unetpp import UNetPlusPlus


class ConvLSTMCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Conv2d(channels*2, channels*4, 3, padding=1)

    def forward(self, x, state=None):
        if state is None:
            h, c = torch.zeros_like(x), torch.zeros_like(x)
        else:
            h, c = state
        i,f,o,g = torch.chunk(self.gates(torch.cat([x,h],dim=1)),4,dim=1)
        i,f,o,g = torch.sigmoid(i),torch.sigmoid(f),torch.sigmoid(o),torch.tanh(g)
        c = f*c + i*g
        h = o*torch.tanh(c)
        return h,c


class ConvLSTMSegmenter(nn.Module):
    def __init__(self, in_channels=3, num_classes=9, base_channels=32):
        super().__init__()
        self.unetpp = UNetPlusPlus(in_channels,num_classes,base_channels)
        self.temporal = ConvLSTMCell(self.unetpp.filters[-1])

    def forward(self, images):
        if images.dim()==4:
            images=images.unsqueeze(1)
        state=None
        current_feats=None
        for t in range(images.shape[1]):
            feats=self.unetpp.encode(images[:,t])
            state=self.temporal(feats[-1],state)
            current_feats=feats
        return self.unetpp.decode(current_feats,bottleneck=state[0])
