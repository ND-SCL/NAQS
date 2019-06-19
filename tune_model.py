import argparse
import torch
import torch.optim as optim
import data
from child_pytorch import get_model
import utility
import backend_pytorch as backend
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
batch_size = 64


def lr_schedule(optimizer, epoch):
    new_lr = lr
    if epoch > 25:
        new_lr = 0.001
    if epoch > 50:
        new_lr = 0.0008
    if epoch > 75:
        new_lr = 0.0005
    if epoch > 100:
        new_lr = 0.0003
    adjust_learning_rate(optimizer, new_lr)
    return new_lr


def adjust_learning_rate(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def tune(paras=[], dataset='CIFAR10'):
    # quantize = True if 'act_num_int_bits' in paras[0] else False
    arch_paras, quan_paras = utility.split_paras(paras)
    input_shape, num_classes = data.get_info(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data = data.get_data(
        name=dataset, device=device,
        shuffle=True, batch_size=batch_size, augment=True)
    model = get_model(
        input_shape, arch_paras, num_classes,
        device=device,
        multi_gpu=args.multi_gpu,
        do_bn=args.do_bn)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True)
    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=lr,
    #     betas=(0.9, 0.999),
    #     eps=1e-7,
    #     weight_decay=0,
    #     amsgrad=False
    #     )
    best_acc = 0
    best_quan_acc = 0
    for epoch in range(1, args.epochs+1):
        epoch_lr = lr_schedule(optimizer, epoch)
        print('-' * 80)
        print(f"Epoch {epoch} \t LR: {epoch_lr}" +
              f"\t Best Acc: {best_acc:6.3%}" +
              (f"quantized: {best_quan_acc:6.3%}" if quan_paras is not None
               else ''))
        print("Training ...")
        running_loss, running_correction, running_total = 0, 0, 0
        bar_width = 30
        start = time.time()
        for input_batch, label_batch in train_data:
            batch_loss, batch_correction = backend.batch_fit(
                model, input_batch, label_batch, optimizer)
            end = time.time()
            running_loss += batch_loss
            running_correction += batch_correction
            running_total += 1
            train_acc = running_correction / running_total / batch_size
            train_loss = running_loss / running_total / batch_size
            epoch_percentage = running_total / len(train_data)
            print('|' + '='*(math.ceil(bar_width * epoch_percentage)-1) +
                  '>' +
                  ' '*(bar_width - math.ceil(bar_width * epoch_percentage)) +
                  '|' + f"{epoch_percentage:4.1%}-{end-start:4.2f}s" +
                  f"\t loss: {train_loss:.5}, acc: {train_acc:6.3%}  ",
                  end=('\r' if epoch_percentage < 1 else '\n'))
        print("Training finished, start evaluating ...")
        running_loss, running_correction, running_total = 0, 0, 0
        start = time.time()
        for input_batch, label_batch in val_data:
            with torch.no_grad():
                batch_loss, batch_correction = backend.batch_fit(
                    model, input_batch, label_batch)
                end = time.time()
                running_loss += batch_loss
                running_correction += batch_correction
                running_total += 1
                val_acc = running_correction / running_total / batch_size
                val_loss = running_loss / running_total / batch_size
                epoch_percentage = running_total / len(val_data)
            print('|' + '='*(math.ceil(bar_width * epoch_percentage)-1) +
                  '>' +
                  ' '*(bar_width - math.ceil(bar_width * epoch_percentage)) +
                  '|' + f"{epoch_percentage:4.1%}-{end-start:4.2f}s" +
                  f"\t loss: {val_loss:.5}, acc: {val_acc:6.3%}  ",
                  end=('\r' if epoch_percentage < 1 else '\n'))
        if val_acc > best_acc:
            best_acc = val_acc
        if quan_paras is not None:
            print("Start evaluating with quantization ...")
            start = time.time()
            for input_batch, label_batch in val_data:
                running_loss, running_correction, running_total = 0, 0, 0
                with torch.no_grad():
                    batch_loss, batch_correction = backend.batch_fit(
                        model, input_batch, label_batch, quan_paras=quan_paras)
                    end = time.time()
                    running_loss += batch_loss
                    running_correction += batch_correction
                    running_total += input_batch.size(0)
                    val_acc = running_correction / running_total
                    val_loss = running_loss / running_total
                    epoch_percentage = running_total / len(train_data)
                print('|' + '='*(math.ceil(bar_width * epoch_percentage)-1) +
                      '>' + ' '*(bar_width - math.ceil(
                        bar_width * epoch_percentage)) + '|' +
                      f"{epoch_percentage:4.1%}-{end-start:4.2f}s" +
                      f"\t val_loss: {val_loss:.5}, val_acc: {val_acc:6.3%}  ",
                      end=('\r' if epoch_percentage < 1 else '\n'))
            if val_acc > best_quan_acc:
                best_quan_acc = val_acc


if __name__ == '__main__':
    tune(model_to_tune.paras, args.dataset)
