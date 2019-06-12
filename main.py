import argparse

import torch

import backend
import child
import data
import controller_nl as ctrl
from config import ARCH_SPACE


def get_args():
    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument(
        '-d', '--dataset',
        default='MNIST',
        help="supported dataset including : 1. MNIST, 2. CIFAR10")
    parser.add_argument(
        '-l', '--layers',
        type=int,
        default=1,
        help="the number of child network layers")
    parser.add_argument(
        '-e', '--epoch',
        type=int,
        default=40,
        help="the total epochs for model fitting")
    return parser.parse_args()


def main():
    args = get_args()
    dataset = args.dataset
    num_layers = args.layers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    train_data, val_data = data.get_data(dataset, device)
    input_shape, num_classes = data.get_info(dataset)
    print(input_shape)
    agent = ctrl.get_agent(ARCH_SPACE, num_layers, device)
    rollout, paras = agent.rollout()
    model, optimizer = child.get_model(input_shape, paras, num_classes, device)
    for cell in model.graph:
        print(cell)
    backend.fit(model, optimizer, train_data, val_data, args.epoch)


if __name__ == '__main__':
    import random
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    main()
