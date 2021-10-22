import torch
import torchinfo
from torchvision import models as models
from torchviz import make_dot
# import os
# os.environ["PATH"]+=os.pathsep+'C:/Program Files/Graphviz/bin/'
from models.efficientnet import EfficientNetB0

# net = models.shufflenet_v2_x1_0()
# net = models.mobilenet_v2()
# net = models.mnasnet1_0()
net = EfficientNetB0()
net.to("cuda:0")

# torchinfo.summary(net, (1, 3, 224, 224))
x = torch.zeros(1, 3, 224, 244)
x = x.to("cuda:0")
# make_dot(net(x), params=dict(list(net.named_parameters()))).render("torchviz", format='png')
print(net)

with open('log_%s.txt', 'w') as f:
    m_info = str(torchinfo.summary(net, (1, 3, 224, 224), verbose=0))
    f.write('%s\n\n' % m_info)