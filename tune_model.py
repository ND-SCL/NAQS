import argparse
import torch
import torch.optim as optim
import data
from child_pytorch import get_model
import utility
import backend_pytorch as backend
import model_to_tune

parser = argparse.ArgumentParser('Parser User Input Arguments')
parser.add_argument(
    '-m', '--multi_gpu',
    actions='store_true',
    help="use more than one gpu, default false"
    )
parser.add_argument(
    '-e', '--epochs',
    type=int,
    help="number of epochs, default is 150"
    )
parser.add_argument(
    '-v', '--verbosity',
    type=int,
    choices=range(3),
    default=0,
    help="verbosity level: 0 (default), 1 and 2 with 2 being the most verbose"
    )
parser.add_argument(
    '-d', '--dataset',
    default='CIFAR10',
    help="supported dataset including : 1. MNIST, 2. CIFAR10 (default)"
    )
args = parser.parse_args()


def lr_schedule(optimizer, epoch):
    if epoch == 15:
        adjust_learning_rate(optimizer, 5e-3)
    if epoch == 25:
        adjust_learning_rate(optimizer, 1e-3)


def adjust_learning_rate(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def tune(paras=[], dataset='CIFAR10'):
    # quantize = True if 'act_num_int_bits' in paras[0] else False
    arch_paras, quan_paras = utility.split_paras(paras)
    input_shape, num_classes = data.get_info(dataset)
    train_data, val_data = data.get_data(
        shuffle=True, batch_size=128, augment=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        input_shape, arch_paras, num_classes,
        device=device,
        multi_gpu=args.multi_gpu,)
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True)
    backend.fit(
        model, optimizer,
        train_data, val_data,
        epochs=args.epochs,
        verbosity=args.verbosity,
        quan_paras=quan_paras,
        lr_schedule=lr_schedule)


if __name__ == '__main__':
    tune(model_to_tune.paras, args.dataset)
