import torch
import torch.nn as nn

net = torch.load("outputs/CLNET/temp.pt")

for name, module in net.named_modules():
    if 'pconv' in name:
        print(name)