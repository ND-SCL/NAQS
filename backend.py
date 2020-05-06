import numpy as np
import time

import torch
import torch.nn.functional as F


def fit(model, optimizer=None, train_data=None, val_data=None, epochs=40,
        verbosity=0, quan_paras=None, lr_schedule=None):
    acc = []
    loss = []
    timer = Timer()
    timer.reset()
    for epoch in range(epochs):
        if lr_schedule is not None:
            lr_schedule(optimizer, epoch)
        if train_data is not None:
            if verbosity > 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: ", end='')
            train_loss, train_acc = epoch_fit(
                model, train_data, optimizer, verbosity=verbosity)
            if train_acc > 0.99:
                break
            if verbosity > 0:
                print(f"Train Loss: {train_loss:.5} - " +
                      f"Train Acc: {train_acc:6.3%} - ",
                      end='')
        if val_data is not None:
            with torch.no_grad():
                val_loss, val_acc = epoch_fit(
                    model, val_data,
                    quan_paras=quan_paras
                    )
            if verbosity > 0:
                print(f" Val Loss: {val_loss:.5} - Val Acc: {val_acc:6.3%} - ",
                      end=' ')
            acc.append(val_acc)
            loss.append(val_loss)
        if verbosity > 0:
            print(f"Elasped time: {timer.sample()}")
        # if epoch == 10 and val_data is not None and val_acc < 0.12:
        #     break
    if len(acc) > 4:
        return np.mean(loss[-5:]), np.mean(acc[-5:])  # train and validate
    elif len(acc) == 0:
        return None, None  # train but not validate
    else:
        return val_loss, val_acc  # just valid


def epoch_fit(model, data, optimizer=None, quan_paras=None, verbosity=0):
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    running_loss, running_correction, running_total = 0, 0, 0
    for input_batch, label_batch in data:
        loss, correction = batch_fit(
            model, input_batch, label_batch, optimizer, quan_paras)
        running_loss += loss
        running_correction += correction
        running_total += input_batch.size(0)
        if verbosity > 1:
            print(f"batch loss: {loss}\t " +
                  f"batch acc: {correction/input_batch.size(0)}")
    return running_loss / running_total, running_correction / running_total


def batch_fit(model, input_batch, label_batch, optimizer=None,
              quan_paras=None):
    output_batch = model(input_batch, quan_paras)
    _, prediction = torch.max(output_batch, 1)
    loss = F.cross_entropy(output_batch, label_batch)
    correction = (prediction == label_batch).sum()
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), correction.item()


def is_convergence(latest_acc):
    if len(latest_acc) < 5:
        return False
    else:
        latest_acc = latest_acc[-5:]
        return not (max(latest_acc) - latest_acc[0] > 0.005)


class Timer():
    def __init__(self):
        self.time = 0

    def reset(self):
        self.time = time.time()

    def sample(self):
        elasped_time = time.time() - self.time
        self.time = time.time()
        return elasped_time


if __name__ == '__main__':
    import data
    import mnist_net
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    model, optimizer = mnist_net.get_model(device)
    train_data, val_data = data.get_data('MNIST', device)
    fit(model, optimizer, train_data, val_data)
