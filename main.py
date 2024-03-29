"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torchinfo import summary

import os
import argparse
import platform

from models import *
from models.clnet_sw import CLNet_V10, CLNet_V11, CLNet_V12_bn
from models.clnet import CLNet_V0
from models.clnetv2 import CLNetV1_C1B1_sw, CLNetV1_C1B2_sw, CLNetV1_C1B3_sw
from utils import progress_bar
import time
import random

random_seed = 1
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # multi-GPU


timestr = time.strftime("%Y%m%d-%H%M%S")
parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
args = parser.parse_args()

curr_os = platform.system()
print("Current OS : %s" % curr_os)

if "Windows" in curr_os:
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif "Darwin" in curr_os:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Model
print("==> Building model..")
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
# net = CLNet_V10(10)
# net = CLNet_V11(10)
# net = CLNet_V12_bn(10)
# net = CLNetV1_C1B1_sw(10)
# net = CLNetV1_C1B2_sw(10)
net = CLNetV1_C1B3_sw(10)

netkey = net.__class__.__name__
# net = net.to("mps")
if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

log_path = "outputs/" + netkey
os.makedirs(log_path, exist_ok=True)

with open(log_path + "/log_%s.txt" % timestr, "w") as f:
    f.write("Networks : %s\n" % netkey)
    m_info = str(summary(net, (1, 3, 32, 32), verbose=0))
    f.write("%s\n\n" % m_info)

# Training
def train(epoch, dir_path=None, fname=None):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )

    with open("outputs/" + dir_path + "/log_%s.txt" % fname, "a") as f:
        f.write(
            "Epoch [%d] |Train| Loss: %.3f, Acc: %.3f \t"
            % (epoch, train_loss / (batch_idx + 1), 100.0 * correct / total)
        )


def test(epoch, dir_path=None, fname=None):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/ckpt.pth")
        best_acc = acc

    with open("outputs/" + dir_path + "/log_%s.txt" % fname, "a") as f:
        f.write("|Test| Loss: %.3f, Acc: %.3f \n" % (test_loss / (batch_idx + 1), acc))


max_epoch = 200
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)
# optimizer = optim.SGD(net.parameters(), lr=1e-2)
# optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-3)
optimizer = optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)


for epoch in range(start_epoch, start_epoch + max_epoch):
    net = net.to(device)
    train(epoch, netkey, timestr)
    test(epoch, netkey, timestr)
    scheduler.step()
