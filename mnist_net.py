import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MnistNet(nn.Module):
    def __init__(self, shape=(1, 28, 28), num_classes=10):
        super(MnistNet, self).__init__()
        self.conv = nn.Conv2d(
            1, 8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
            )
        self.fc = nn.Linear(8 * 28 * 28, num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return self.fc(x.view(x.shape[0], -1))


def get_model(device=torch.device('cpu')):
    model = MnistNet().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=True
    )
    return model, optimizer


if __name__ == '__main__':
    from backend import trainer
