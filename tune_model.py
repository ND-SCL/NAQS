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
    action='store_true',
    help="use more than one gpu, default false"
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


def lr_schedule(optimizer, epoch):
    new_lr = lr if epoch < 100 else lr / 10
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
        shuffle=True, batch_size=128, augment=True)
    model = get_model(
        input_shape, arch_paras, num_classes,
        device=device,
        multi_gpu=args.multi_gpu,)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True)
    best_acc = 0
    best_quan_acc = 0
    for epoch in range(args.epochs):
        epoch_lr = lr_schedule(optimizer, epoch)
        print('=' * 80)
        print(f"Epoch {epoch} \t\t\t\t\t\t\t lr: {epoch_lr}" +
              f"\t best acc ever: {best_acc:6.3%}" +
              (f"quantized: {best_quan_acc:6.3%}" if quan_paras is not None
               else ''))
        print("Training ...")
        running_loss, running_correction, running_total = 0, 0, 0
        for input_batch, label_batch in train_data:
            batch_loss, batch_correction = backend.batch_fit(
                model, input_batch, label_batch, optimizer)
            running_loss += batch_loss
            running_correction += batch_correction
            running_total += input_batch.size(0)
            train_acc = running_correction / running_total
            train_loss = running_loss / running_total
            epoch_percentage = running_total / len(train_data)
            print('\r', '=' * int(100 * epoch_percentage) +
                  '>' if epoch_percentage < 0.99 else '=' +
                  f'\t {epoch_percentage*100:.2%}' +
                  f"loss: {train_loss:.5}, acc: {train_acc:6.3%}",
                  end='' if epoch_percentage < 0.99 else '\n')
        print("Training finished, start evaluating ...")
        for input_batch, label_batch in val_data:
            running_loss, running_correction, running_total = 0, 0, 0
            with torch.no_grad():
                batch_loss, batch_correction = backend.batch_fit(
                    model, input_batch, label_batch)
                running_loss += batch_loss
                running_correction += batch_correction
                running_total += input_batch.size(0)
                val_acc = running_correction / running_total
                val_loss = running_loss / running_total
                epoch_percentage = running_total / len(train_data)
                print('\r', '=' * int(100 * epoch_percentage) +
                      '>' if epoch_percentage < 0.99 else '=' +
                      f'\t {epoch_percentage:6.3%}' +
                      f"loss: {val_loss:.5}, acc: {val_acc:6.3%}",
                      end='' if epoch_percentage < 0.99 else '\n')
        if val_acc > best_acc:
            best_acc = val_acc
        if quan_paras is not None:
            print("Start evaluating with quantization ...")
            for input_batch, label_batch in val_data:
                running_loss, running_correction, running_total = 0, 0, 0
                with torch.no_grad():
                    batch_loss, batch_correction = backend.batch_fit(
                        model, input_batch, label_batch, quan_paras=quan_paras)
                    running_loss += batch_loss
                    running_correction += batch_correction
                    running_total += input_batch.size(0)
                    val_acc = running_correction / running_total
                    val_loss = running_loss / running_total
                    epoch_percentage = running_total / len(train_data)
                    print('\r', '=' * int(100 * epoch_percentage) +
                          '>' if epoch_percentage < 0.99 else '=' +
                          f'\t {epoch_percentage*:6.3%}' +
                          f"loss: {val_loss:.5}, acc: {val_acc:6.3%}",
                          end='' if epoch_percentage < 0.99 else '\n')
            if val_acc > best_quan_acc:
                best_quan_acc = val_acc


if __name__ == '__main__':
    tune(model_to_tune.paras, args.dataset)
