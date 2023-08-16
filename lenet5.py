import os

import cv2
import torch
import torch.nn as nn
import numpy as np


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x, target=None):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        if target is not None:
            loss = nn.CrossEntropyLoss()(x, target)
            return x, loss
        return x


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


def load_data(data_path, is_train):
    data = np.loadtxt(data_path, dtype=str, delimiter=',')
    if is_train:
        data = data[1:, :]
        y, x = data[:, 0], data[:, 1:]
        return x, y
    else:
        x = data[1:, :]
        return x


def augment_data(x, y):
    kernel = np.ones((2, 2), np.uint8)
    augment_1 = [cv2.erode(i.reshape(28, 28).astype(np.uint8), kernel, iterations=1) for i in x]
    augment_1 = np.array(augment_1)
    augment_1 = augment_1.reshape(-1, 28 * 28)
    augment_2 = []
    for i in augment_1:
        new_im = i.copy()
        new_im[new_im > 20] = 255
        augment_2.append(new_im)
    augment_2 = np.array(augment_2)

    x = np.concatenate((x, augment_1, augment_2), axis=0)
    y = np.concatenate((y, y, y), axis=0)
    return x, y


def train_val_split(x, y, partition=0.8):
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    part = int(len(x) * partition)
    x_train = x[:part, :]
    y_train = y[:part]
    x_val = x[part:, :]
    y_val = y[part:]

    return x_train, y_train, x_val, y_val


def eval(net, split):
    if split == 'train':
        x, y = x_train, y_train
    else:
        x, y = x_val, y_val
    net.eval()
    correct = 0
    for i in range(0, len(x), batch_size):
        x_batch, y_batch = x[i:i + batch_size], y[i:i + batch_size]
        x_batch = torch.from_numpy(x_batch.astype(np.float32)).view(-1, 1, 28, 28)
        y_batch = torch.from_numpy(y_batch.astype(np.int64))
        logits = net(x_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
    return correct / len(x)


def get_batch(split):
    if split == 'train':
        x, y = x_train, y_train
    else:
        x, y = x_val, y_val
    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]


if __name__ == '__main__':
    data_path = 'data/train.csv'
    x, y = load_data(data_path, is_train=True)
    x, y = augment_data(x, y)
    x_train, y_train, x_val, y_val = train_val_split(x, y)

    print("Load data done")

    lr_decay = 0.5
    lr = 0.001
    batch_size = 64
    epoch = 4
    eval_iter = 100

    net = HandmadeLeNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for e in range(epoch):
        for i, (x_batch, y_batch) in enumerate(get_batch('train')):
            x_batch = torch.from_numpy(x_batch.astype(np.float32)).view(-1, 1, 28, 28)
            y_batch = torch.from_numpy(y_batch.astype(np.int64))
            logits, loss = net(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % eval_iter == 0:
                print('epoch: {}, iter: {}, loss: {:.4f}, train_acc: {:.4f}, val_acc: {:.4f}'.format(
                    e, i, loss.item(), eval(net, 'train'), eval(net, 'val')))
                net.train()
                torch.save(net.state_dict(), 'lenet5_new_ft.pth')
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
