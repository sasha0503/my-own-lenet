import os

from datetime import datetime

import tqdm
import cv2
import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=60, kernel_size=7)
        self.conv4 = nn.Conv2d(in_channels=60, out_channels=120, kernel_size=5)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)
        self.batch_norm_1 = nn.BatchNorm2d(1)
        self.batch_norm_2 = nn.BatchNorm2d(6)
        self.batch_norm_3 = nn.BatchNorm2d(16)
        self.batch_norm_4 = nn.BatchNorm2d(60)
        self.batch_norm_5 = nn.BatchNorm2d(120)
        self.dropout_1 = nn.Dropout(0.5)

    def forward(self, x, target=None):
        # input 32x32
        x = self.relu(self.conv1(x))
        # output 30x30
        x = self.pool(x)
        # output 15x15
        x = self.relu(self.conv2(x))
        # output 13x13
        x = self.pool_2(x)
        # output 12x12
        x = self.relu(self.conv3(x))
        # output 6x6
        x = self.pool_2(x)
        # output 5x5
        x = self.relu(self.conv4(x))
        # output 1x1

        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout_1(x)
        x = self.fc2(x)
        if target is not None:
            loss = nn.CrossEntropyLoss()(x, target)
            return x, loss
        return x


def load_data(data_path, is_train, load_augmented=False):
    train_path = os.path.join(data_path, 'train.csv')
    test_path = os.path.join(data_path, 'test.csv')
    augmented_paths = [os.path.join(data_path, 'augment_1.csv'), os.path.join(data_path, 'augment_2.csv'),
                       os.path.join(data_path, 'augment_3_0.csv'), os.path.join(data_path, 'augment_3_1.csv')]
    if is_train:
        data = np.loadtxt(train_path, dtype=str, delimiter=',')
        data = data[1:, :]
        if load_augmented:
            for path in augmented_paths:
                new_data = np.loadtxt(path, dtype=str, delimiter=',')
                new_data = new_data[1:, :]
                data = np.concatenate((data, new_data), axis=0)
        y, x = data[:, 0], data[:, 1:]
        return x, y
    else:
        data = np.loadtxt(test_path, dtype=str, delimiter=',')
        x = data[1:, :]
        return x


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

    # evaluate only 10 random batches
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    x = x[:batch_size * 10]
    y = y[:batch_size * 10]

    net.eval()
    correct = 0
    for i in range(0, len(x), batch_size):
        x_batch, y_batch = x[i:i + batch_size], y[i:i + batch_size]
        x_batch = torch.from_numpy(x_batch.astype(np.float32)).view(-1, 1, 28, 28)
        y_batch = torch.from_numpy(y_batch.astype(np.int64))
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
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
    model_name = f'lenet5_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'
    data_path = 'data'

    print("Loading data...")
    x, y = load_data(data_path, is_train=True, load_augmented=True)
    x_train, y_train, x_val, y_val = train_val_split(x, y)

    print("Load data done")
    print(f"Train size: {len(x_train)}")

    # save 100 samples from train data
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    im2save = x[indices[-100:]]
    im2save = im2save.reshape(-1, 28, 28)
    os.makedirs('samples', exist_ok=True)
    for i, im in enumerate(im2save):
        cv2.imwrite(f'samples/{i}.png', im.astype(np.float32))

    lr_decay = 0.5
    lr = 0.001
    batch_size = 64
    epoch = 6
    eval_iter = 100

    net = LeNet5()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    best_val = 0

    for e in range(epoch):
        for i, (x_batch, y_batch) in enumerate(get_batch('train')):
            x_batch = torch.from_numpy(x_batch.astype(np.float32)).view(-1, 1, 28, 28)
            y_batch = torch.from_numpy(y_batch.astype(np.int64))
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits, loss = net(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % eval_iter == 0:
                val_acc = eval(net, 'val')
                train_acc = eval(net, 'train')
                print('epoch: {}, iter: {}, loss: {:.4f}, train_acc: {:.4f}, val_acc: {:.4f}'.format(
                    e, i, loss.item(), train_acc, val_acc))
                net.train()
                if val_acc > best_val:
                    best_val = val_acc
                    torch.save(net.state_dict(), f'models/{model_name}')
        lr *= lr_decay ** e
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
