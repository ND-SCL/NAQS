import torch
import torch.nn.functional as F


def fit(model, optimizer, train_data, val_data=None, epochs=40,
        verbose=True):
    for epoch in range(epochs):
        loss, acc = epoch_fit(model, train_data, optimizer)
        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs}, " +
                  f"Train Loss: {loss}, Train Acc: {acc:6.3%}", end='')
        if val_data is not None:
            loss, acc = epoch_fit(model, val_data)
            if verbose:
                print(f" Val Loss: {loss}, Val Acc: {acc:6.3%}")


def epoch_fit(model, data, optimizer=None):
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    running_loss, running_correction, running_total = 0, 0, 0
    for input_batch, label_batch in data:
        loss, correction = batch_fit(
            model, input_batch, label_batch, optimizer)
        running_loss += loss
        running_correction += correction
        running_total += input_batch.size(0)
    return running_loss / running_total, running_correction / running_total


def batch_fit(model, input_batch, label_batch, optimizer=None):
    output_batch = model(input_batch)
    _, prediction = torch.max(output_batch, 1)
    # print(prediction.shape, label_batch.shape)
    loss = F.cross_entropy(output_batch, label_batch)
    correction = (prediction == label_batch).sum()
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), correction.item()


# def trainer(net, validate=True):
#     # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.0)
#     optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=True)
#     # optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
#     # net = net.to(device)
#     epochs = 150
#     for epoch in range(epochs):
#         running_loss = 0
#         running_correction = 0
#         running_total = 0
#         net.train()
#         for data in trainloader:
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = net(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             running_correction += (predicted == labels).sum().item()
#             running_total += images.size(0)
#         if validate:
#             val_loss, val_acc =  validator(net)
#             print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {running_loss/running_total*batch_size}, Acc: {running_correction/running_total:6.3%}, Val loss: {val_loss}, Val acc: {val_acc:6.3%}")
#         else:
#             print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {running_loss/running_total*batch_size}, Acc: {running_correction/running_total:6.3%}")


# def validator(net):
#     # net = net.to(device)
#     net.eval()
#     running_loss = 0
#     running_correction = 0
#     running_total = 0
#     for data in validloader:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         outputs = net(images)
#         loss = criterion(outputs, labels)
#         running_loss += float(loss)
#         logits = F.softmax(outputs, dim=1)
#         _, predicted = torch.max(logits, 1)
#         running_correction += (predicted == labels).sum().item()
#         running_total += images.size(0)
#     return running_loss/running_total*batch_size,\
#         running_correction/running_total


if __name__ == '__main__':
    import data
    import mnist_net
    model, optimizer = mnist_net.get_model()
    train_data, val_data = data.get_data('MNIST')
    fit(model, optimizer, train_data, val_data)
