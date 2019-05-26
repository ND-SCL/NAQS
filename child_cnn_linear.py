import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_backend import device, trainer

# print(f"using device {device}")
def build_layers(input_shape, num_classes, arch_paras):
    conv_layers = []
    pool_layers = []
    drop_layers = []
    in_channels, height, width = input_shape
    drop_rate = 0.2
    for paras in arch_paras:
        out_channels = paras['num_filters']
        # define convolutional layers
        conv_layer = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(paras['filter_height'], paras['filter_width']),
            stride=(paras['stride_height'], paras['stride_width']),
            padding=(1, 1))
        conv_layers.append(conv_layer)
        # define max pooling parameters
        if paras['pool_size'] > 1:
            pool_layer = nn.MaxPool2d(paras['pool_size'], padding=(1, 1))
            drop_layer = nn.Dropout(p=drop_rate)
            drop_rate += 0.1
        else:
            pool_layer = None
            drop_layer = None
        pool_layers.append(pool_layer)
        drop_layers.append(drop_layer)
        in_channels = out_channels
    num_features = compute_num_flat_features(input_shape, conv_layers, pool_layers, drop_layers)
    fc_layers = []
    fc_layer = nn.Linear(num_features, num_classes)
    fc_layers.append(fc_layer)
    return conv_layers, pool_layers, drop_layers, fc_layers


def compute_num_flat_features(input_shape, conv_layers, pool_layers, drop_layers):
    x = torch.randn((1,) + input_shape)
    x = conv_forward(x, conv_layers, pool_layers, drop_layers)
    size = x.size()
    return size[1] * size[2] * size[3]


def conv_forward(x, conv_layers, pool_layers, drop_layers):
    for i in range(len(conv_layers)):
        x = F.relu(conv_layers[i](x))
        if pool_layers[i]:
            x = pool_layers[i](x)
        if drop_layers[i]:
            x = drop_layers[i](x)
    return x


def fc_forward(x, fc_layers):
    for j in range(len(fc_layers)):
        x = fc_layers[j](x)
        if j < len(fc_layers) - 1:
            x = F.relu(x)
        else:
            pass
    return x


class CNN(nn.Module):
    def __init__(self, conv_layers, pool_layers, drop_layers, fc_layers):
        super(CNN, self).__init__()
        self.conv_layers = conv_layers
        self.pool_layers = pool_layers
        self.drop_layers = drop_layers
        self.fc_layers = fc_layers
        for i in range(len(conv_layers)):
            setattr(self, 'conv{}'.format(i+1), pool_layers[i])
            # setattr(self, f'conv{i+1}', conv_layers[i])
        for i in range(len(pool_layers)):
            if pool_layers[i]:
                setattr(self, 'pool{}'.format(i+1), pool_layers[i])
        for i in range(len(drop_layers)):
            if drop_layers[i]:
                setattr(self, 'drop{}'.format(i+1), drop_layers[i])
        for j in range(len(fc_layers)):
            setattr(self, 'fc{}'.format(j+1), fc_layers[j])

    def forward(self, x):
        x = conv_forward(x, self.conv_layers, self.pool_layers, self.drop_layers)
        x = x.view(x.size()[0], -1)
        x = fc_forward(x, self.fc_layers)
        return x


class ChildCNN(object):
	def __init__(self, input_shape, num_classes):
		self.input_shape = input_shape
		self.num_classes = num_classes

	def update_arch(self, arch_paras=[]):
		conv_layers, pool_layers, drop_layers, fc_layers = build_layers(self.input_shape, self.num_classes, arch_paras)
		model = CNN(conv_layers, pool_layers, drop_layers, fc_layers)
		# model = nn.DataParallel(model)
		self.model = model.to(device)

	def train(self):
		trainer(self.model)

def compute_padding(height, width, kernel_height, kernel_width, stride_height, stride_width):
    num_height_strides = math.floor((height - 1) / stride_height)
    num_height_paddings = num_height_strides * stride_height + kernel_height - height
    num_width_strides = math.floor((width - 1) / stride_width)
    num_width_paddings = num_width_strides * stride_width + kernel_width - width
    return math.floor(num_height_paddings / 2), math.ceil(num_height_paddings / 2),\
        math.floor(num_width_paddings / 2), math.ceil(num_width_paddings / 2)





if __name__ == '__main__':
	arch_paras = [
		{'num_filters': 32, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1}, 
		{'num_filters': 32, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 2}, 
		{'num_filters': 64, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1}, 
		{'num_filters': 64, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 2}, 
		{'num_filters': 128, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 1}, 
		{'num_filters': 128, 'filter_height': 3, 'filter_width': 3, 'stride_height': 1, 'stride_width': 1, 'pool_size': 2}]

	input_shape = (3, 32, 32)
	num_classes = 10
	child_cnn = ChildCNN(input_shape, num_classes)
	child_cnn.update_arch(arch_paras)
	child_cnn.train()



		
		

