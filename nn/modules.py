import torch.nn as nn
import torch


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, channels_in, channels_out, kernel=1, stride=1, padding=None, groups=1, dilation=1,
                 activation=True):
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel, stride, autopad(kernel, padding, dilation), dilation,
                              groups, bias=False)
        self.bn = nn.BatchNorm2d(channels_out)
        self.act = self.default_act if activation is True else activation if isinstance(activation,
                                                                                        nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, channels_in, channels_out, shortcut=True, groups=1, kernels=(3, 3), expand=0.5):
        super().__init__()
        c_ = int(channels_out * expand)  # hidden channels
        self.cv1 = Conv(channels_in, c_, kernels[0], 1)
        self.cv2 = Conv(c_, channels_out, kernels[1], 1, groups=groups)
        self.add = shortcut and channels_in == channels_out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, channels_in, channels_out, number=1, shortcut=True, groups=1, expand=0.5):
        super().__init__()
        self.c = int(channels_out * expand)  # hidden channels
        self.cv1 = Conv(channels_in, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + number) * self.c, channels_out, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, groups, kernels=((3, 3), (3, 3,)), expand=1.0) for _ in range(number))

    # def forward(self, x):
    #     y = list(self.cv1(x).chunk(2, 1))
    #     y.extend(m(y[-1]) for m in self.m)
    #     return self.cv2(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Classify(nn.Module):
    def __init__(self, channels_in, channels_out, kernel=1, stride=1, padding=None, groups=1):
        super().__init__()
        c_ = 1280  # efficient_b0 size (1280)
        self.conv = Conv(channels_in, c_, kernel, stride, autopad(kernel, padding), groups)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, channels_out)  # to x(b, channel_out)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)
