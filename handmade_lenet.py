import torch
import torch.nn as nn


class HandmadeLeNet:
    def __init__(self):
        self.conv1 = torch.randn(6, 1, 5, 5, requires_grad=True)
        nn.init.kaiming_normal_(self.conv1, mode='fan_in', nonlinearity='relu')
        self.conv2 = torch.randn(16, 6, 5, 5, requires_grad=True)
        nn.init.kaiming_normal_(self.conv2, mode='fan_in', nonlinearity='relu')
        self.conv3 = torch.randn(120, 16, 5, 5, requires_grad=True)
        nn.init.kaiming_normal_(self.conv3, mode='fan_in', nonlinearity='relu')
        self.fc1 = torch.randn(84, 120, requires_grad=True)
        nn.init.kaiming_normal_(self.fc1, mode='fan_in', nonlinearity='relu')
        self.fc2 = torch.randn(10, 84, requires_grad=True)
        nn.init.kaiming_normal_(self.fc2, mode='fan_in', nonlinearity='relu')

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        return [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]

    def train(self):
        for param in self.parameters():
            param.requires_grad = True

    def eval(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, target=None):
        x = torch.nn.functional.pad(x, (2, 2, 2, 2))
        x = torch.nn.functional.conv2d(x, self.conv1)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.conv2d(x, self.conv2)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.conv2d(x, self.conv3)
        x = x.view(x.shape[0], -1)
        x = torch.nn.functional.linear(x, self.fc1)
        x = torch.nn.functional.linear(x, self.fc2)
        if target is not None:
            loss = nn.CrossEntropyLoss()(x, target)
            return x, loss
        return x

    def state_dict(self):
        return {'conv1': self.conv1, 'conv2': self.conv2, 'conv3': self.conv3, 'fc1': self.fc1, 'fc2': self.fc2}
