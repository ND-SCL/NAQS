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
    normalize = Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    # normalize = Normalize(
    #     (0.47359734773635864, 0.47359734773635864, 0.47359734773635864),
    #     (0.2515689432621002, 0.2515689432621002, 0.2515689432621002))
    # normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if augment is True:
        train_transform = transforms.Compose([
            transforms.RandomAffine(10, translate=(0.07, 0.07)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])
    trainloader = DataLoader(
        datasets.CIFAR10(
            root='./data/CIFAR10',
            train=True,
            download=True,
            transform=train_transform
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        # pin_memory=True
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
        num_workers=4,
        # pin_memory=True
    )
    return trainloader, valloader


def get_imagenet(shuffle=True, batch_size=64, augment=False):
    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # normalize = transforms.Normalize(
    #     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if augment is True:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            normalize
            ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
        ])
    # train_transform = val_transform
    trainloader = DataLoader(
        datasets.ImageFolder(
            root='/ImageNet/train',
            transform=train_transform
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
    )
    valloader = DataLoader(
        datasets.ImageFolder(
            root='/ImageNet/val',
            transform=val_transform
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
    )
    return trainloader, valloader


def get_tiny_imagenet(shuffle=True, batch_size=64, augment=False):
    # normalize = transforms.Normalize(
    #     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    normalize = transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if augment is True:
        train_transform = transforms.Compose([
            transforms.RandomAffine(10, translate=(0.07, 0.07)),
            transforms.ToTensor(),
            normalize
            ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    # train_transform = val_transform
    trainloader = DataLoader(
        datasets.ImageFolder(
            root='./data/TinyImageNet/train',
            transform=train_transform
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
    )
    valloader = DataLoader(
        datasets.ImageFolder(
            root='./data/TinyImageNet/val',
            transform=val_transform
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
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
    'ImageNet': {
        'generator': get_imagenet,
        'shape': (3, 224, 224),
        'num_classes': 10
        },
    'TinyImageNet': {
        'generator': get_tiny_imagenet,
        'shape': (3, 64, 64),
        'num_classes': 200
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
    return WrappedDataLoader(trainloader, device), \
        WrappedDataLoader(valloader, device)


def get_info(name='CIFAR10'):
    return DATA[name]['shape'], DATA[name]['num_classes']


class Normalize():
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        if self.mean is None:
            mean = x.mean([1, 2], True)
        else:
            mean = torch.tensor(
                self.mean).unsqueeze(-1).unsqueeze(-1).expand_as(x).type_as(x)
        if self.std is None:
            std = x.std(1, True, True).std(2, True, True)
        else:
            std = torch.tensor(
                self.std).unsqueeze(-1).unsqueeze(-1).expand_as(x).type_as(x)
        return (x - mean) / std


def get_mean():
    trainloader = DataLoader(
        datasets.CIFAR10(
            root='./data/CIFAR10',
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=50000,
        shuffle=False,
        num_workers=2)
    for batch, _ in trainloader:
        # print(batch[0][0])
        pass
    # return batch.mean([0, 2, 3]), \
    #     (batch[:, 0].std(), batch[:, 1].std(), batch[:, 2].std())
    return batch.mean().item(), batch.std().item()


if __name__ == '__main__':
    print(get_mean())
