from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_mnist():
    mnist_transform = transforms.Compose([
        transforms.ToTensor()])
    trainloader = DataLoader(
        datasets.MNIST(
            root='./data/MNIST',
            train=True,
            download=True,
            transform=mnist_transform
        ),
        batch_size=64,
        shuffle=True,
        num_workers=2
    )
    valloader = DataLoader(
        datasets.MNIST(
            root='./data/MNIST',
            train=False,
            download=True,
            transform=mnist_transform
        ),
        batch_size=64,
        shuffle=False,
        num_workers=2
    )
    return trainloader, valloader


def get_cifar10():
    cifar10_transform = transforms.Compose([
        transforms.ToTensor()])
    trainloader = DataLoader(
        datasets.CIFAR10(
            root='./data/CIFAR10',
            train=True,
            download=True,
            transform=cifar10_transform
        ),
        batch_size=64,
        shuffle=True,
        num_workers=2
    )
    valloader = DataLoader(
        datasets.CIFAR10(
            root='./data/CIFAR10',
            train=False,
            download=True,
            transform=cifar10_transform
        ),
        batch_size=64,
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


def get_data(name='MNIST'):
    return DATA[name]['generator']()


def get_data_info(name='CIFAR10'):
    return DATA[name]['shape'], DATA[name]['num_classes']
