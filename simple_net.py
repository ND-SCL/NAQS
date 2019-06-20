import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv_pad_1 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv_1 = nn.Conv2d(3, 32, (3, 3), (1, 1))
        self.pool_pad_1 = nn.ZeroPad2d((0, 0, 0, 0))
        self.pool_1 = nn.MaxPool2d(1)
        self.drop_1 = nn.Dropout(p=0.2)
        self.conv_pad_2 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv_2 = nn.Conv2d(32, 32, (3, 3), (1, 1))
        self.pool_pad_2 = nn.ZeroPad2d((0, 0, 0, 0))
        self.pool_2 = nn.MaxPool2d(2)
        self.drop_2 = nn.Dropout(p=0.2)
        self.conv_pad_3 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv_3 = nn.Conv2d(32, 64, (3, 3), (1, 1))
        self.pool_pad_3 = nn.ZeroPad2d((0, 0, 0, 0))
        self.pool_3 = nn.MaxPool2d(1)
        self.drop_3 = nn.Dropout(p=0.2)
        self.conv_pad_4 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv_4 = nn.Conv2d(64, 64, (3, 3), (1, 1))
        self.pool_pad_4 = nn.ZeroPad2d((0, 0, 0, 0))
        self.pool_4 = nn.MaxPool2d(2)
        self.drop_4 = nn.Dropout(p=0.2)
        self.conv_pad_5 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv_5 = nn.Conv2d(64, 128, (3, 3), (1, 1))
        self.pool_pad_5 = nn.ZeroPad2d((0, 0, 0, 0))
        self.pool_5 = nn.MaxPool2d(1)
        self.drop_5 = nn.Dropout(p=0.2)
        self.conv_pad_6 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv_6 = nn.Conv2d(128, 128, (3, 3), (1, 1))
        self.pool_pad_6 = nn.ZeroPad2d((0, 0, 0, 0))
        self.pool_6 = nn.MaxPool2d(2)
        self.drop_6 = nn.Dropout(p=0.2)
        self.fc = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x, y=None):
        x = self.conv_pad_1(x)
        x = F.relu(self.conv_1(x))
        x = self.pool_1(self.pool_pad_1(x))
        x = self.drop_1(x)
        x = self.conv_pad_2(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_2(self.pool_pad_2(x))
        x = self.drop_2(x)
        x = self.conv_pad_3(x)
        x = F.relu(self.conv_3(x))
        x = self.pool_3(self.pool_pad_3(x))
        x = self.drop_3(x)
        x = self.conv_pad_4(x)
        x = F.relu(self.conv_4(x))
        x = self.pool_4(self.pool_pad_4(x))
        x = self.drop_4(x)
        x = self.conv_pad_5(x)
        x = F.relu(self.conv_5(x))
        x = self.pool_5(self.pool_pad_5(x))
        x = self.drop_5(x)
        x = self.conv_pad_6(x)
        x = F.relu(self.conv_6(x))
        x = self.pool_6(self.pool_pad_6(x))
        x = self.drop_6(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    import data
    import backend_pytorch as backend
    import time

    dataset = 'CIFAR10'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data = data.get_data(
        dataset, device, shuffle=True, batch_size=128)
    input_shape, num_classes = data.get_info(dataset)
    model = SimpleNet().to(device)
    if device.type == 'cuda':
        print("using parallel data")
        model = torch.nn.DataParallel(model)
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True)
    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=0.001,
    #     betas=(0.9, 0.999),
    #     eps=1e-8,
    #     weight_decay=0.0,
    #     amsgrad=True
    # )
    start = time.time()
    backend.fit(
        model, optimizer,
        train_data, val_data,
        epochs=200,
        verbosity=1
        )
    end = time.time()
    print(end-start)
