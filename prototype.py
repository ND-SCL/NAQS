import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_backend import device, trainer 

print(f"Using device {device}")
torch.manual_seed(0)


class CNN(nn.Module):

	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.conv2 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.pool2 = nn.MaxPool2d((2, 2))
		self.drop2 = nn.Dropout(p=0.4)
		self.conv3 = nn.Conv2d(32, 64, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.conv4 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.pool4 = nn.MaxPool2d((2, 2))
		self.drop4 = nn.Dropout(p=0.4)
		self.conv5 = nn.Conv2d(64, 128, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.conv6 = nn.Conv2d(128, 128, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.pool6 = nn.MaxPool2d((2, 2))
		self.drop6 = nn.Dropout(p=0.4)
		self.fc1 = nn.Linear(128 * 4 * 4, 128)
		self.fc2 = nn.Linear(128, 10)
		# for i in range(1, 6):
		# 	conv_layer = getattr(self, f'conv{i}')
		# 	nn.init.xavier_uniform_(conv_layer.weight)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		x = self.drop2(x)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.pool4(x)
		x = self.drop4(x)
		x = F.relu(self.conv5(x))
		x = F.relu(self.conv6(x))
		x = self.pool6(x)
		x = self.drop6(x)
		x = x.view(x.size()[0], -1)	
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

class VGG16(nn.Module):

	def __init__(self):
		super(VGG16, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.conv2 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.pool2 = nn.MaxPool2d((2, 2))
		self.drop2 = nn.Dropout(p=0.2)
		self.conv3 = nn.Conv2d(64, 128, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.conv4 = nn.Conv2d(128, 128, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.pool4 = nn.MaxPool2d((2, 2))
		self.drop4 = nn.Dropout(p=0.3)
		self.conv5 = nn.Conv2d(128, 256, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.conv6 = nn.Conv2d(256, 256, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.conv7 = nn.Conv2d(256, 256, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.pool7 = nn.MaxPool2d((2, 2))
		self.drop7 = nn.Dropout(p=0.4)
		self.conv8 = nn.Conv2d(256, 512, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.conv9 = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.conv10 = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.pool10 = nn.MaxPool2d((2, 2))
		self.drop10 = nn.Dropout(p=0.4)
		self.conv11 = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.conv12 = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.conv13 = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=(1, 1), padding=(1, 1))
		self.pool13 = nn.MaxPool2d((2, 2))
		self.drop13 = nn.Dropout(p=0.4)
		self.fc = nn.Linear(512 * 1 * 1, 10)
		# for i in range(1, 10):
		# 	conv_layer = getattr(self, f'conv{i}')
		# 	nn.init.xavier_uniform_(conv_layer.weight)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		# x = self.drop2(x)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.pool4(x)
		# x = self.drop4(x)
		x = F.relu(self.conv5(x))
		x = F.relu(self.conv6(x))
		x = F.relu(self.conv7(x))
		x = self.pool7(x)
		# x = self.drop7(x)
		x = F.relu(self.conv8(x))
		x = F.relu(self.conv9(x))
		# x = F.relu(self.conv10(x))
		x = self.pool10(x)
		# x = self.drop10(x)
		# x = F.relu(self.conv11(x))
		# x = F.relu(self.conv12(x))
		# x = F.relu(self.conv13(x))
		x = self.pool13(x)
		# x = self.drop13(x)

		x = x.view(x.size()[0], -1)	
		x = self.fc(x)
		return x	


if __name__ == '__main__':
	net = CNN().to(device)
	# inputs = torch.randn((1, 3, 32, 32)).to(device)
	# outputs = net(inputs)
	# print(outputs.size())
	trainer(net)




