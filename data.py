import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF


def get_mnist(shuffle=True, batch_size=64, augment=False):
    mnist_transform = transforms.Compose([
        transforms.ToTensor()])
    trainloader = DataLoader(
        datasets.MNIST(
            root='./data/MNIST', train=True, download=True,
            transform=mnist_transform
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )
    valloader = DataLoader(
        datasets.MNIST(
            root='./data/MNIST',
            train=False,
            download=True,
            transform=mnist_transform
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    return trainloader, valloader


def get_cifar10(shuffle=True, batch_size=64, augment=False):
    plain_transform = [
        transforms.ToTensor(),
        transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                )]
    if augment:
        transform = [
            transforms.Resize(size=(64, 64)),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            plain_transform[0],
            plain_transform[1]
            ]
        train_transform = transforms.Compose(transform)
    else:
        train_transform = transforms.Compose(plain_transform)
    val_transform = transforms.Compose(plain_transform)
    trainloader = DataLoader(
        datasets.CIFAR10(
            root='./data/CIFAR10',
            train=True,
            download=True,
            transform=train_transform
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )
    valloader = DataLoader(
        datasets.CIFAR10(
            root='./data/CIFAR10',
            train=False,
            download=True,
            transform=val_transform
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    return trainloader, valloader


def get_hymenoptera_data():
    hymenoptera_data_transform = transforms.Compose([
        transforms.ToTensor()])
    trainloader = DataLoader(
        datasets.ImageFolder(
            root='./data/hymenoptera_data/train',
            transform=hymenoptera_data_transform
        ),
        batch_size=1,
        shuffle=True,
        num_workers=2
    )
    valloader = DataLoader(
        datasets.ImageFolder(
            root='./data/hymenoptera_data/val',
            transform=hymenoptera_data_transform
        ),
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    return trainloader, valloader


DATA = {
    'MNIST': {
        'generator': get_mnist,
        'shape': (1, 28, 28),
        'num_classes': 10
        },
    'CIFAR10': {
        'generator': get_cifar10,
        'shape': (3, 32, 32),
        'num_classes': 10
        },
    'hymenoptera_data': {
        'generator': get_hymenoptera_data,
        'shape': (),
        'num_classes': 2
        }
}


class WrappedDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for b in self.dataloader:
            yield [v.to(self.device) for v in b]


def get_data(name='MNIST', device=torch.device('cpu'), shuffle=True,
             batch_size=64, augment=False):
    trainloader, valloader = DATA[name]['generator'](
        shuffle, batch_size, augment=augment)
    # return dataloader
    return WrappedDataLoader(trainloader, device), \
        WrappedDataLoader(valloader, device)


def get_info(name='CIFAR10'):
    return DATA[name]['shape'], DATA[name]['num_classes']
