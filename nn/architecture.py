from nn.modules import Conv, C2f, Classify
import torch.nn as nn


class YoloClassifySmall(nn.Module):  # YOLOv8s-cls
    def __init__(self, nc):
        super().__init__()
        self.conv1 = Conv(3, 80, (3, 3), (2, 2), (1, 1))

        self.conv2 = Conv(80, 160, (3, 3), (2, 2), (1, 1))
        self.c2f2 = C2f(160, 160, 3)

        self.conv3 = Conv(160, 320, (3, 3), (2, 2), (1, 1))
        self.c2f3 = C2f(320, 320, 6)

        self.conv4 = Conv(320, 640, (3, 3), (2, 2), (1, 1))
        self.c2f4 = C2f(640, 640, 6)

        self.conv5 = Conv(640, 1280, (3, 3), (2, 2), (1, 1))
        self.c2f5 = C2f(1280, 1280, 3)

        self.classify = Classify(1280, nc, (1, 1), (1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.c2f2(self.conv2(out))
        out = self.c2f3(self.conv3(out))
        out = self.c2f4(self.conv4(out))
        out = self.c2f5(self.conv5(out))
        out = self.classify(out)

        return out


class MyNet(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.conv1 = Conv(3, 80, (3, 3), (2, 2), (1, 1))

        self.conv2 = Conv(80, 160, (3, 3), (2, 2), (1, 1))
        self.c2f2 = C2f(160, 160, 3)

        self.conv3 = Conv(160, 160, (3, 3), (2, 2), (1, 1))
        self.c2f3 = C2f(160, 160, 6)

        self.conv4 = Conv(160, 320, (3, 3), (2, 2), (1, 1))
        self.c2f4 = C2f(320, 320, 6)

        self.classify = Classify(320, nc, (1, 1), (1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.c2f2(self.conv2(out))
        out = self.c2f3(self.conv3(out))
        out = self.c2f4(self.conv4(out))
        out = self.classify(out)

        return out


class MyNet2(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.conv1 = Conv(3, 80, (3, 3), (2, 2), (1, 1))

        self.conv2 = Conv(80, 160, (3, 3), (2, 2), (1, 1))
        self.c2f2 = C2f(160, 160, 3)

        self.classify = Classify(160, nc, (1, 1), (1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.c2f2(self.conv2(out))
        out = self.classify(out)

        return out

