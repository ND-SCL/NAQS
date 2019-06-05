import torch

import argparse
import data
import mnist_net


def get_args():
    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument(
        '-d', '--dataset',
        default='MNIST',
        help="supported dataset including : 1. MNIST, 2. CIFAR10")
    return parser.parse_args()


def main():
    args = get_args()
    dataset = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    model, optimizer = mnist_net.get_model(device)
    backend.fit(model, optimizer, train_data, val_data)

    model = mnist_net.get_model(dataset, device)
