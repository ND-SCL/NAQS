import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
data_utilization = 1


transform_list = []
# transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
# transform_list.append(transforms.RandomRotation(0))
transform_list.append(transforms.ToTensor())

transform = transforms.Compose(transform_list)

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform)

trainset, validset = torch.utils.data.random_split(trainset, [45000, 5000])
train_size = int(len(trainset) * data_utilization)
trainset, _ = torch.utils.data.random_split(
    trainset,
    [train_size, len(trainset) - train_size])

# _, validset = torch.utils.data.random_split(trainset, [45000, 5000])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()


def trainer(net, validate=True):
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.0)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=True)
    # optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
    # net = net.to(device)
    epochs = 150
    for epoch in range(epochs):
        running_loss = 0
        running_correction = 0
        running_total = 0
        net.train()
        for data in trainloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_correction += (predicted == labels).sum().item()
            running_total += images.size(0)
        if validate:
            val_loss, val_acc =  validator(net)
            print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {running_loss/running_total*batch_size}, Acc: {running_correction/running_total:6.3%}, Val loss: {val_loss}, Val acc: {val_acc:6.3%}")
        else:
            print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {running_loss/running_total*batch_size}, Acc: {running_correction/running_total:6.3%}")


def validator(net):
    # net = net.to(device)
    net.eval()
    running_loss = 0
    running_correction = 0
    running_total = 0
    for data in validloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        running_loss += float(loss)
        logits = F.softmax(outputs, dim=1)
        _, predicted = torch.max(logits, 1)
        running_correction += (predicted == labels).sum().item()
        running_total += images.size(0)
    return running_loss/running_total*batch_size,\
        running_correction/running_total


if __name__ == '__main__':
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images[0][0])
