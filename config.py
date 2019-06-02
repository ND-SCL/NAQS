# import torch
import torchvision.datasets as datasets


class HymenopteraData(datasets.ImageFolder):
    def __init__(self, root, train=True, download=False, transform=[]):
        if train:
            self.root = root + '/train'
        else:
            self.root = root + 'val'
        datasets.ImageFolder.__init__(
            self,
            root=root,
            transform=transform)

# class hymenoptera(datasets):
#     def __init__(self):
#         pass

#     def __getitem__(self):
#         pass

#     def __len__(self):
#         pass

DATA = {
    'MNIST': (datasets.MNIST, (1, 28, 28)),
    'CIFAR10': (datasets.CIFAR10, (3, 32, 32)),
    'hymenoptera_data': (HymenopteraData, ())
}
