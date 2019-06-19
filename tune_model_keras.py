import argparse
import torch
import torch.optim as optim
import data
from child_keras import ChildCNN
import utility
import backend_keras as backend
import model_to_tune
import time
import math

parser = argparse.ArgumentParser('Parser User Input Arguments')
parser.add_argument(
    '-m', '--multi_gpu',
    action='store_true',
    help="use more than one gpu, default false"
    )
parser.add_argument(
    '-db', '--do_bn',
    action='store_true',
    help="use batch normalization, default is false"
    )
parser.add_argument(
    '-e', '--epochs',
    type=int,
    default=150,
    help="number of epochs, default is 150"
    )
parser.add_argument(
    '-v', '--verbosity',
    type=int,
    choices=range(3),
    default=1,
    help="verbosity level: 0, 1 and 2 (default) with 2 being the most verbose"
    )
parser.add_argument(
    '-d', '--dataset',
    default='CIFAR10',
    help="supported dataset including : 1. MNIST, 2. CIFAR10 (default)"
    )
args = parser.parse_args()


lr = 0.01
batch_size = 128




def adjust_learning_rate(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def tune(paras=[], dataset='CIFAR10'):
    # quantize = True if 'act_num_int_bits' in paras[0] else False
    arch_paras, quan_paras = utility.split_paras(paras)
    input_shape, num_classes = data.get_info(dataset)
    model = ChildCNN(arch_paras, quan_paras, dataset)
    model.train(args.epochs, batch_size=128, validate=True, verbose=1)



if __name__ == '__main__':
    tune(model_to_tune.paras, args.dataset)
