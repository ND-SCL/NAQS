import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data
import backend


drop_rate = 0.2

drop_rates = [0, 0.2, 0, 0.3, 0, 0.4, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0.5,
              0, 0, 0.5]


class Cell():
    def __init__(self, id):
        self.id = id
        self.prev = []
        self.in_channels = 0
        self.output_shape = ()
        self.conv_pad = []
        self.used = False
        self.conv = None
        self.pool = None
        self.drop = None
        self.bn = None
        self.para_num = 0

    def __repr__(self):
        return f"id: {self.id} " + \
            f"prev {self.prev} " + \
            f"in_channels: {self.in_channels} " + \
            f"output_shape : {self.output_shape} " + \
            f"conv_padding: {self.conv_pad} " + \
            f"conv: {self.conv} " + \
            f"pool_padding: {self.pool_pad} " + \
            f"pool: {self.pool} " + \
            f"drop: {self.drop} " + \
            f"used: {self.used}\n"


def build_graph(input_shape, arch_paras):
    graph = []
    cell_id = 0
    last_output_shape = input_shape
    for layer_paras in arch_paras:
        cell = Cell(cell_id)
        num_filters = layer_paras['num_filters']
        filter_height = layer_paras['filter_height']
        filter_width = layer_paras['filter_width']
        stride_height = layer_paras['stride_height'] \
            if 'stride_height' in layer_paras else 1
        stride_width = layer_paras['stride_width'] \
            if 'stride_width' in layer_paras else 1
        pool_size = layer_paras['pool_size'] \
            if 'pool_size' in layer_paras else 1
        pool_stride = pool_size
        cell.prev = [cell.id - 1]
        in_channels, in_height, in_width = last_output_shape
        if 'anchor_point' in layer_paras:
            anchor_point = layer_paras['anchor_point']
            out_height, out_width = 0, 0
            for l in range(len(anchor_point)):
                if anchor_point[l] == 1:
                    cell.prev.append(l)
                    in_channels += graph[l].output_shape[0]
                    in_height = max(
                        in_height, graph[l].output_shape[1]
                        )
                    in_width = max(
                        in_width, graph[l].output_shape[2]
                        )
        for p in cell.prev:
            padding_height, out_height = compute_padding(
                in_height,
                filter_height,
                stride_height
                )
            padding_width, out_width = compute_padding(
                in_width,
                filter_width,
                stride_width
                )
            prev_output_shape = input_shape if p < 0 else graph[p].output_shape
            cell.conv_pad.append((
                math.floor(
                    (in_width + padding_width -
                        prev_output_shape[2])/2),
                math.ceil(
                    (in_width + padding_width -
                        prev_output_shape[2])/2),
                math.floor(
                    (in_height + padding_height -
                        prev_output_shape[1])/2),
                math.ceil(
                    (in_height + padding_height -
                        prev_output_shape[1])/2)
                    )
                )
        # else:
        #     padding_height, out_height = compute_padding(
        #                 in_height,
        #                 filter_height,
        #                 stride_height
        #                 )
        #     padding_width, out_width = compute_padding(
        #                 in_width,
        #                 filter_width,
        #                 stride_width
        #                 )
        #     cell.conv_pad = [(
        #             math.floor(padding_width/2),
        #             math.ceil(padding_width/2),
        #             math.floor(padding_height/2),
        #             math.ceil(padding_height/2))]
        cell.in_channels = in_channels
        cell.conv = nn.Conv2d(
            cell.in_channels, num_filters,
            kernel_size=(filter_height, filter_width),
            stride=(stride_height, stride_width)
            )
        cell.bn = nn.BatchNorm2d(num_filters)
        padding_height, out_height = compute_padding(
                    out_height,
                    pool_size,
                    pool_stride,
                    )
        padding_width, out_width = compute_padding(
                    out_width,
                    pool_size,
                    pool_stride,
                    )
        cell.output_shape = (num_filters, out_height, out_width)
        cell.pool_pad = (
                    math.floor(padding_width/2),
                    math.ceil(padding_width/2),
                    math.floor(padding_height/2),
                    math.ceil(padding_height/2))
        cell.pool = nn.MaxPool2d(pool_size)
        cell.drop = nn.Dropout(p=drop_rates[cell.id])
        cell.para_num = (in_channels * filter_height * filter_width + 1
                         ) * num_filters
        graph.append(cell)
        cell_id += 1
        last_output_shape = cell.output_shape
    return graph


def compute_padding(input_size, kernel_size, stride):
    output_size = math.floor((input_size + stride - 1) / stride)
    padding = max(0, (output_size-1) * stride + kernel_size - input_size)
    return padding, output_size


class CNN(nn.Module):
    def __init__(self, graph, num_classes, do_bn=False):
        super(CNN, self).__init__()
        self.graph = graph
        self.do_bn = do_bn
        for cell in self.graph:
            for l, p in zip(cell.prev, cell.conv_pad):
                setattr(self, 'conv_pad_{}_{}'.format(cell.id, l), p)
            setattr(self, 'conv_{}'.format(cell.id), cell.conv)
            setattr(self, 'pool_pad_{}'.format(cell.id), cell.pool_pad)
            setattr(self, 'pool_{}'.format(cell.id), cell.pool)
            setattr(self, 'drop_{}'.format(cell.id), cell.drop)
            if do_bn is True:
                setattr(self, 'bn_{}'.format(cell.id), cell.bn)
        self.num_features = compute_num_features(self.graph[-1].output_shape)
        self.fc1 = nn.Linear(self.num_features, 256)
        self.fc_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, quan_paras=None):
        if quan_paras is not None:
            x = quantize(x, 8, 8, signed=True)
        output = [x]
        for cell in self.graph:
            padded_input = []
            for l in cell.prev:
                conv_pad = getattr(
                    self, 'conv_pad_{}_{}'.format(cell.id, l))
                x = F.pad(output[l+1], conv_pad)
                padded_input.append(x)
            x = torch.cat(padded_input, dim=1)
            conv = getattr(self, 'conv_{}'.format(cell.id))
            pool_pad = getattr(self, 'pool_pad_{}'.format(cell.id))
            pool = getattr(self, 'pool_{}'.format(cell.id))
            drop = getattr(self, 'drop_{}'.format(cell.id))
            if quan_paras is not None:
                weight, bias = conv.weight, conv.bias
                weight = quantize(
                            weight,
                            quan_paras[cell.id]['weight_num_int_bits'],
                            quan_paras[cell.id]['weight_num_frac_bits'],
                            signed=True)
                bias = quantize(
                        bias,
                        quan_paras[cell.id]['weight_num_int_bits'],
                        quan_paras[cell.id]['weight_num_frac_bits'],
                        signed=True
                        )
                stride = conv.stride
                x = F.conv2d(x, weight, bias, stride=stride)
                x = quantize(
                    x,
                    quan_paras[cell.id]['act_num_int_bits'],
                    quan_paras[cell.id]['act_num_frac_bits'],
                    signed=False
                    )
            else:
                x = conv(x)
            x = F.relu(x)
            if self.do_bn is True:
                bn = getattr(self, 'bn_{}'.format(cell.id))
                x = bn(x)
            x = pool(F.pad(x, pool_pad))
            x = drop(x)
            output.append(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.do_bn is True:
            x = self.fc_bn(x)
        x = nn.Dropout(p=0.5)(x)
        x = self.fc2(x)
        return x

    def quantize_weight(self, num_int_bits, num_frac_bits):
        for cell in self.graph:
            conv = getattr(self, 'conv_{}'.format(cell.id))
            conv.weight = quantize(conv.wieght, num_int_bits, num_frac_bits)
            conv.bias = quantize(conv.bias, num_int_bits, num_frac_bits)


def compute_num_features(shape):
    return shape[-1] * shape[-2] * shape[-3]


def quantize(x, num_int_bits, num_frac_bits, signed=True):
    precision = 1 / 2 ** num_frac_bits
    x = torch.round(x / precision) * precision
    if signed is True:
        bound = 2 ** (num_int_bits - 1)
        return torch.clamp(x, -bound, bound-precision)
    else:
        bound = 2 ** num_int_bits
        return torch.clamp(x, 0, bound-precision)


def get_model(input_shape, paras, num_classes, device=torch.device('cpu'),
              multi_gpu=False, do_bn=False):
    graph = build_graph(input_shape, paras)
    model = CNN(graph, num_classes, do_bn=do_bn)
    if device.type != 'cpu' and multi_gpu is True:
        print("using parallel data")
        model = torch.nn.DataParallel(model)
    return model.to(device), get_optimizer(model, 'SGD')


def get_optimizer(model, name='SGD'):
    adam_optim = optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=0,
        amsgrad=False
        )
    sgd_optim = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True)
    if name == 'SGD':
        return sgd_optim
    else:
        return adam_optim


class ChildCNN(object):
    def __init__(self, arch_paras=None, quan_paras=None, dataset='CIFAR10'):
        self.input_shape, self.num_classes = data.get_info(dataset)
        self.dataset = dataset
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            )
        if arch_paras is not None:
            self.model = get_model(
                self.input_shape, arch_paras, self.num_classes, self.device)
            self.optimizer = get_optimizer(self.model, 'SGD')
        if quan_paras is not None:
            self.quan_paras = quan_paras

    def update_architecture(self, arch_paras=None):
        if arch_paras is not None:
            # print("updating architecture")
            self.model = get_model(
                self.input_shape, arch_paras, self.num_classes, self.device)
            self.optimizer = get_optimizer(self.model, 'SGD')

    def update_quantization(self, quan_paras=None):
        if quan_paras is not None:
            self.quan_paras = quan_paras

    def fit(self, validate=False, quantize=False, verbosity=0, epochs=40):
        train_data, val_data = data.get_data(
            self.dataset, self.device,
            shuffle=True,
            batch_size=128,
            augment=True)
        loss, acc = backend.fit(
            self.model, self.optimizer,
            train_data=train_data,
            val_data=None if validate is False else val_data,
            epochs=epochs,
            verbosity=verbosity,
            quan_paras=None if quantize is False else self.quan_paras)
        return loss, acc

    def train(self, batch_size=128, epochs=40,
              verbosity=True, validate=False):
        train_data, val_data = data.get_data(
            self.dataset, self.device,
            shuffle=True,
            batch_size=batch_size,
            augment=True)
        acc = backend.fit(
            self.model, self.optimizer,
            train_data=train_data,
            val_data=None if validate is False else val_data,
            epochs=epochs,
            verbosity=verbosity
            )
        return acc

    def validate(self, quantize=False, verbosity=False):
        _, val_data = data.get_data(
            self.dataset, self.device,
            shuffle=False,
            batch_size=128)
        acc = backend.fit(
            self.model, self.optimizer,
            val_data=val_data,
            quan_paras=self.quan_paras,
            epochs=1,
            verbosity=verbosity)
        return acc

    def collect_garbage(self):
        del self.model


if __name__ == '__main__':
    import controller_nl
    from config import ARCH_SPACE
    import random
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)

    input_shape = (3, 32, 32)
    num_classes = 10
    # for l in paras:
    #     print(l)
    # graph = build_graph(input_shape, paras)
    # for cell in graph:
    #     print(cell)
    arch_paras = [
        {'anchor_point': [], 'filter_height': 5, 'filter_width': 5,
         'num_filters': 64, 'pool_size': 2,
         'act_num_int_bits': 2, 'act_num_frac_bits': 4,
         'weight_num_int_bits': 2, 'weight_num_frac_bits': 6},
        {'anchor_point': [], 'filter_height': 7, 'filter_width': 5,
         'num_filters': 36, 'pool_size': 2,
         'act_num_int_bits': 2, 'act_num_frac_bits': 1,
         'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
        {'anchor_point': [0], 'filter_height': 7, 'filter_width': 3,
         'num_filters': 64, 'pool_size': 2,
         'act_num_int_bits': 2, 'act_num_frac_bits': 5,
         'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
        {'anchor_point': [0, 0], 'filter_height': 1, 'filter_width': 7,
         'num_filters': 48, 'pool_size': 2,
         'act_num_int_bits': 1, 'act_num_frac_bits': 3,
         'weight_num_int_bits': 3, 'weight_num_frac_bits': 6},
        {'anchor_point': [1, 0, 1], 'filter_height': 1, 'filter_width': 1,
         'num_filters': 64, 'pool_size': 2,
         'act_num_int_bits': 2, 'act_num_frac_bits': 4,
         'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
        {'anchor_point': [0, 0, 0, 0], 'filter_height': 1, 'filter_width': 1,
         'num_filters': 24, 'pool_size': 2,
         'act_num_int_bits': 2, 'act_num_frac_bits': 3,
         'weight_num_int_bits': 0, 'weight_num_frac_bits': 5}]
    model, optimizer = get_model(
                input_shape, arch_paras, num_classes)
    x = torch.randn(1, *input_shape)
    print(model.graph)
    # model = CNN(input_shape, paras, num_classes)
    # input = torch.randn(5, *input_shape)
    # print(input.shape)
    # output = model(input)
    # print(output, output.shape)
    # graph = model.graph
    # for cell in graph:
    #     print(cell)

