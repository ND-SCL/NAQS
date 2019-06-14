import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


drop_rate = 0.2


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

    def __repr__(self):
        return f"id: {self.id} " + \
            f"prev {self.prev} " + \
            f"in_channels: {self.in_channels} " + \
            f"output_shape : {self.output_shape} " + \
            f"padding: {self.conv_pad} " + \
            f"conv: {self.conv} " + \
            f"pool: {self.pool} " + \
            f"drop: {self.drop} " + \
            f"used: {self.used}\n"


def build_graph(input_shape, arch_paras):
    graph = []
    cell_id = 0
    prev_output_shape = input_shape
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
        if 'anchor_point' in layer_paras:
            anchor_point = layer_paras['anchor_point']
            in_channels, in_height, in_width = 0, 0, 0
            out_height, out_width = 0, 0
            if cell_id == len(arch_paras) - 1:
                for i in range(cell_id):
                    if graph[i].used is False:
                        anchor_point[i] = 1
                        # graph[i].used = True
            for l in range(len(anchor_point)):
                if anchor_point[l] == 1:
                    graph[l].used = True
                    cell.prev.append(l)
                    in_channels += graph[l].output_shape[0]
                    in_height = max(
                        in_height, graph[l].output_shape[1]
                        )
                    in_width = max(
                        in_width, graph[l].output_shape[2]
                        )
            if cell.prev:
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
                    cell.conv_pad.append(nn.ZeroPad2d((
                        math.floor(
                            (in_width + padding_width -
                                graph[p].output_shape[2])/2),
                        math.ceil(
                            (in_width + padding_width -
                                graph[p].output_shape[2])/2),
                        math.floor(
                            (in_height + padding_height -
                                graph[p].output_shape[1])/2),
                        math.ceil(
                            (in_height + padding_height -
                                graph[p].output_shape[1])/2)
                            ))
                        )
            else:
                cell.prev.append(-1)
                in_channels = input_shape[0]
                # print(input_shape)
                out_height = math.ceil(
                    (input_shape[1] - filter_height) / stride_height) + 1
                out_width = math.ceil(
                    (input_shape[2] - filter_width) / stride_width) + 1
                padding_height, out_height = compute_padding(
                        input_shape[1],
                        filter_height,
                        stride_height
                        )
                padding_width, out_width = compute_padding(
                        input_shape[2],
                        filter_width,
                        stride_width
                        )
                cell.conv_pad = [nn.ZeroPad2d((
                    math.floor(padding_width/2),
                    math.ceil(padding_width/2),
                    math.floor(padding_height/2),
                    math.ceil(padding_height/2))
                    )]
        else:
            cell.prev = [cell.id - 1]
            in_channels, in_height, in_width = prev_output_shape
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
            cell.conv_pad = [nn.ZeroPad2d((
                    math.floor(padding_width/2),
                    math.ceil(padding_width/2),
                    math.floor(padding_height/2),
                    math.ceil(padding_height/2))
                    )]

        # print(cell.in_channels)
        # print(in_channels)
        cell.in_channels = in_channels
        cell.conv = nn.Conv2d(
            cell.in_channels, num_filters,
            kernel_size=(filter_height, filter_width),
            stride=(stride_height, stride_width)
            )
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
        # print("out_height: ", out_height)
        cell.output_shape = (num_filters, out_height, out_width)
        cell.pool_pad = nn.ZeroPad2d((
                    math.floor(padding_width/2),
                    math.ceil(padding_width/2),
                    math.floor(padding_height/2),
                    math.ceil(padding_height/2))
                    )
        cell.pool = nn.MaxPool2d(pool_size)
        cell.drop = nn.Dropout(p=drop_rate)
        graph.append(cell)
        cell_id += 1
        prev_output_shape = cell.output_shape
    return graph


def compute_padding(input_size, kernel_size, stride):
    output_size = math.floor((input_size + stride - 1) / stride)
    padding = max(0, (output_size-1) * stride + kernel_size - input_size)
    return padding, output_size


class CNN(nn.Module):
    def __init__(self, input_shape, paras, num_classes):
        super(CNN, self).__init__()
        self.graph = build_graph(input_shape, paras)
        for cell in self.graph:
            for l, p in zip(cell.prev, cell.conv_pad):
                setattr(self, 'conv_pad_{}_{}'.format(cell.id, l), p)
            setattr(self, 'conv_{}'.format(cell.id), cell.conv)
            setattr(self, 'pool_pad_{}'.format(cell.id), cell.pool_pad)
            setattr(self, 'pool_{}'.format(cell.id), cell.pool)
            setattr(self, 'drop_{}'.format(cell.id), cell.drop)
        self.num_features = compute_num_features(self.graph[-1].output_shape)
        self.fc = nn.Linear(self.num_features, num_classes)
        # for cell in self.graph:
        #     print(cell)

    def forward(self, x, quan_paras=None):
        output = []
        image = x
        for cell in self.graph:
            # print(cell.id)
            if cell.prev[0] == -1:
                padding = getattr(self, 'conv_pad_{}_{}'.format(cell.id, -1))
                x = padding(image)
            else:
                padded_input = []
                for l in cell.prev:
                    conv_pad = getattr(
                        self, 'conv_pad_{}_{}'.format(cell.id, l))
                    x = conv_pad(output[l])
                    padded_input.append(x)
                # for input in padded_input:
                    # print(input.shape)
                x = torch.cat(padded_input, dim=1)
                # print("output shape after concatenation: ", x.shape)
            conv = getattr(self, 'conv_{}'.format(cell.id))
            pool_pad = getattr(self, 'pool_pad_{}'.format(cell.id))
            pool = getattr(self, 'pool_{}'.format(cell.id))
            drop = getattr(self, 'drop_{}'.format(cell.id))
            if quan_paras is not None:
                weight, bias = conv.weight, conv.bias
                conv.weight = nn.Parameter(
                    quantize(
                        weight,
                        quan_paras[cell.id]['weight_num_int_bits'],
                        quan_paras[cell.id]['weight_num_frac_bits']
                        )
                    )
                conv.bias = nn.Parameter(
                    quantize(
                        bias,
                        quan_paras[cell.id]['weight_num_int_bits'],
                        quan_paras[cell.id]['weight_num_frac_bits']
                        )
                    )
            x = F.relu(conv(x))
            if quan_paras is not None:
                conv.weight, conv.bias = weight, bias
                x = quantize(
                    x,
                    quan_paras[cell.id]['act_num_int_bits'],
                    quan_paras[cell.id]['act_num_frac_bits']
                    )
            x = pool(pool_pad(x))
            x = drop(x)
            output.append(x)
        x = x.view(x.size()[0], self.num_features)
        return self.fc(x)

    def quantize_weight(self, num_int_bits, num_frac_bits):
        for cell in self.graph:
            conv = getattr(self, 'conv_{}'.format(cell.id))
            conv.weight = quantize(conv.wieght, num_int_bits, num_frac_bits)
            conv.bias = quantize(conv.bias, num_int_bits, num_frac_bits)


def compute_num_features(shape):
    return shape[-1] * shape[-2] * shape[-3]


def quantize(x, num_int_bits, num_frac_bits, sign=True):
    precision = 1 / 2 ** num_frac_bits
    x = torch.round(x / precision) * precision
    if sign is True:
        bound = 2 ** (num_int_bits - 1)
        return torch.clamp(x, -bound, bound-precision)
    else:
        bound = 2 ** num_int_bits
        return torch.clamp(x, 0, bound-precision)


def get_model(input_shape, paras, num_classes,
              device=torch.device('cpu'), lr=0.001):
    model = CNN(input_shape, paras, num_classes).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=True
    )
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=0.01,
    #     momentum=0.9,
    #     weight_decay=1e-4,
    #     nesterov=True)
    return model, optimizer


if __name__ == '__main__':
    import controller_nl
    from config import ARCH_SPACE
    import random
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)

    num_layers = 10
    agent = controller_nl.Agent(ARCH_SPACE, num_layers)
    rollout, paras = agent.rollout()
    input_shape = (3, 32, 32)
    num_classes = 10
    # for l in paras:
    #     print(l)
    # graph = build_graph(input_shape, paras)
    # for cell in graph:
    #     print(cell)
    model = CNN(input_shape, paras, num_classes)
    # input = torch.randn(5, *input_shape)
    # print(input.shape)
    # output = model(input)
    # print(output, output.shape)
    graph = model.graph
    for cell in graph:
        print(cell)
    input = torch.randn(1, *input_shape)
    output = model(input, None)
    loss = output.sum()-1
    # loss.backward()
    for p in model.parameters():
        if p.grad is None:
            print('!')
            break
