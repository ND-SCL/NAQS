import argparse
import time

import child_pytorch as cp
import child_keras as ck

import torch

parser = argparse.ArgumentParser('Parser User Input Arguments')
parser.add_argument(
    '-f', '--framework',
    default='keras',
    choices=['pytorch', 'keras'],
    help="framework: 'keras' or 'pytorch"
    )
parser.add_argument(
    '-e', '--epochs',
    type=int,
    default=40,
    help="epochs for training"
    )
framework = parser.parse_args().framework
epochs = parser.parse_args().epochs
torch.manual_seed(0)


def remove_anchor_point(arch_paras):
    for layer in arch_paras:
        if 'anchor_point' in layer:
            layer.pop('anchor_point')


arch_paras = [
    {'num_filters': 32, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 1, 'anchor_point': []},
    {'num_filters': 32, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 2, 'anchor_point': [1]},
    {'num_filters': 64, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 1, 'anchor_point': [0, 1]},
    {'num_filters': 64, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 2, 'anchor_point': [0, 0, 1]},
    {'num_filters': 128, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 1, 'anchor_point': [0, 0, 0, 1]},
    {'num_filters': 128, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 2, 'anchor_point': [0, 0, 0, 0, 1]}]
arch_paras = [{'num_filters': 24, 'filter_height': 3, 'filter_width': 5,
              'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
              {'num_filters': 48, 'filter_height': 3, 'filter_width': 5,
              'stride_height': 1, 'stride_width': 2, 'pool_size': 1},
              {'num_filters': 48, 'filter_height': 5, 'filter_width': 3,
              'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
              {'num_filters': 48, 'filter_height': 7, 'filter_width': 7,
              'stride_height': 1, 'stride_width': 1, 'pool_size': 1},
              {'num_filters': 48, 'filter_height': 5, 'filter_width': 7,
              'stride_height': 2, 'stride_width': 1, 'pool_size': 2},
              {'num_filters': 48, 'filter_height': 5, 'filter_width': 7,
              'stride_height': 1, 'stride_width': 1, 'pool_size': 1}]
# remove_anchor_point(arch_paras)
quan_paras = []
for l in range(len(arch_paras)):
    layer = {}
    layer['act_num_int_bits'] = 1
    layer['act_num_frac_bits'] = 4
    layer['weight_num_int_bits'] = 1
    layer['weight_num_frac_bits'] = 6
    quan_paras.append(layer)

dataset = 'CIFAR10'
input_shape = (3, 32, 32)
num_classes = 10

mod = cp if framework == 'pytorch' else ck
child_cnn = mod.ChildCNN(dataset=dataset)
child_cnn.update_architecture(arch_paras=arch_paras)
start = time.time()
acc = child_cnn.train(
    epochs=epochs,
    batch_size=128,
    validate=True,
    verbosity=1)
end = time.time()
print(f"Time: {end-start}", f"Acc: {acc}")
child_cnn.update_quantization(quan_paras=quan_paras)
loss, acc = child_cnn.validate(verbosity=True, quantize=1)
print(f"after quantization, loss: {loss}, acc: {acc}")
