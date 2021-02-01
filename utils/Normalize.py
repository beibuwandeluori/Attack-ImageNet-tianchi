import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i])/self.std[i]

        return x


class Resize(nn.Module):

    def __init__(self, input_size=[224, 224]):
        super(Resize, self).__init__()
        self.input_size = input_size

    def forward(self, input):
        x = F.interpolate(input, size=self.input_size, mode='bilinear', align_corners=True)

        return x


class Permute(nn.Module):
    def __init__(self, permutation=[2,1,0]):
        super().__init__()
        self.permutation = permutation

    def forward(self, input):
        
        return input[:, self.permutation]
