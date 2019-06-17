import logging
import math
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from backend_keras import QuanLayer, trainer, validator, get_info, fit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = logging.getLogger(__name__)
drop_rates = [0, 0.2, 0, 0.3, 0, 0.4, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0.5]
weight_decay = 5e-4


class ChildCNN(object):
    def __init__(self, arch_paras=None, quan_paras=None,
                 dataset='CIFAR10'):
        self.input_shape, self.num_classes = get_info(dataset)
        self.max_act_num_frac_bits = 8
        self.arch_paras = None
        self.quan_paras = None
        if arch_paras:
            self.update_architecture(arch_paras=arch_paras)
        if quan_paras:
            self.update_quantization(quan_paras=quan_paras)

    def update_architecture(self, arch_paras=[]):
        if arch_paras:
            self.arch_paras = arch_paras
            self.build_graph()
            self.model = self.build_model()
            self.num_layers = len(arch_paras)

    def update_quantization(self, quan_paras=[]):
        if quan_paras:
            self.quan_paras = quan_paras
            self.quan_model = self.build_quan_model()

    def build_graph(self):
        self.graph = build_graph(self.input_shape, self.arch_paras)
        # print(self.graph)

    def build_model(self):
        inputs = keras.Input(shape=self.input_shape)
        setattr(self, 'output_'.format(0), inputs)
        x = inputs
        for i in range(len(self.graph)):
            cell = self.graph[i]
            input_group = []
            for l, p in zip(cell.prev, cell.conv_pad):
                prev_output = getattr(self, 'output_'.format(l+1))
                # print(i, l, prev_output.shape)
                if p is not None:
                    x = p(prev_output)
                else:
                    x = prev_output
                # print("after padding: ", x. shape)
                input_group.append(x)
            if len(input_group) > 1:
                x = keras.layers.concatenate(input_group, axis=-1)
            elif len(input_group) == 0:
                x = inputs
            else:
                x = input_group[-1]
            x = cell.conv(x)
            x = keras.layers.Activation('relu')(x)
            x = cell.pool(x)
            if drop_rates[i] > 0:
                x = keras.layers.Dropout(
                    drop_rates[i], name='drop_{}'.format(cell.id))(x)
            setattr(self, 'output_'.format(cell.id+1), x)
        x = keras.layers.Flatten()(x)
        self.dense_layer = keras.layers.Dense(
            self.num_classes, activation='softmax', name='dense')
        predictions = self.dense_layer(x)
        return keras.Model(inputs=inputs, outputs=predictions)

    def build_quan_model(self):
        inputs = keras.Input(shape=self.input_shape)
        initial_quan = QuanLayer(
            num_int_bits=8,
            num_frac_bits=8,
            signed=False,
            name='initial_quan')
        x = initial_quan(inputs)
        setattr(self, 'output_'.format(0), x)
        for i in range(len(self.graph)):
            cell = self.graph[i]
            input_group = []
            for l, p in zip(cell.prev, cell.conv_pad):
                prev_output = getattr(self, 'output_'.format(l+1))
                if p is not None:
                    x = p(prev_output)
                else:
                    x = prev_output
                input_group.append(x)
            if len(input_group) > 1:
                x = keras.layers.concatenate(input_group, axis=-1)
            elif len(input_group) == 0:
                x = inputs
            else:
                x = input_group[-1]
            x = cell.conv(x)
            quan_layer = QuanLayer(
                num_int_bits=self.quan_paras[i]['act_num_int_bits'],
                num_frac_bits=self.quan_paras[i]['act_num_frac_bits'],
                signed=False,
                name='quan_{}'.format(i))
            x = quan_layer(x)
            x = keras.layers.Activation('relu')(x)
            x = cell.pool(x)
            setattr(self, 'output_'.format(cell.id+1), x)
        x = keras.layers.Flatten()(x)
        predictions = self.dense_layer(x)
        return keras.Model(inputs=inputs, outputs=predictions)

    def collect_garbage(self):
        self.backup_weights()
        keras.backend.clear_session()
        self.build_graph()
        self.model = self.build_model()
        if self.quan_paras is not None:
            self.quan_model = self.build_quan_model()
        self.retrieve_weights()

    def backup_weights(self):
        self.conv_weights = []
        for cell in self.graph:
            self.conv_weights.append(cell.conv.get_weights())
        self.dense_weights = self.dense_layer.get_weights()

    def retrieve_weights(self):
        for cell in self.graph:
            cell.conv.set_weights(self.conv_weights[cell.id])
        self.dense_layer.set_weights(self.dense_weights)

    def quantize_paras(self):
        for i in range(self.num_layers):
            layer = self.quan_model.get_layer(name="conv_{}".format(i))
            (w, b) = layer.get_weights()
            bound = 2 ** (self.quan_paras[i]['weight_num_int_bits'] - 1)
            precision = 1 / 2 ** (self.quan_paras[i]['weight_num_frac_bits'])
            w_quantized = np.clip(
                np.around(w / precision) * precision,
                -bound,
                bound - precision)
            b_quantized = np.clip(
                np.around(b / precision) * precision,
                -bound,
                bound - precision)
            layer.set_weights((w_quantized, b_quantized))

    def fit(self, validate=False, quantize=False, verbosity=0, epochs=40):
        loss, acc = fit(
            self.model,
            validate=validate,
            quantize=quantize,
            verbosity=verbosity,
            epochs=epochs)
        return loss, acc

    def train(self, epochs=40, batch_size=128, validate=False, verbosity=0):
        trainer(
            self.model,
            epochs=epochs,
            batch_size=batch_size,
            validate=validate,
            verbosity=verbosity)
        return

    def validate(self, quantize=True, verbosity=0):
        if quantize is False:
            loss, acc = validator(self.model, verbosity=verbosity)
            if verbosity:
                logger.info(
                    f"Validating without quantization, " +
                    f"Loss: {loss:7.5f}, Acc: {acc:6.3%}")
        else:
            self.backup_weights()
            self.quantize_paras()
            loss, acc = validator(
                self.quan_model,
                verbosity=verbosity)
            if verbosity is True:
                print(
                    f"Validating with quantization, " +
                    f"Loss: {loss:7.5f}, Acc: {acc:6.3%}")
            self.retrieve_weights()
        return loss, acc

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)


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
            f"conv_padding: {self.conv_pad} " + \
            f"conv: {self.conv} " + \
            f"pool_padding: {self.pool_pad} " + \
            f"pool: {self.pool} " + \
            f"drop: {self.drop} " + \
            f"used: {self.used}\n"


def build_graph(input_shape, arch_paras):
    graph = []
    cell_id = 0
    input_shape = (input_shape[2], input_shape[0], input_shape[1])
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
                    _, out_height = compute_padding(
                        in_height,
                        filter_height,
                        stride_height
                        )
                    _, out_width = compute_padding(
                        in_width,
                        filter_width,
                        stride_width
                        )
                    top = math.floor((in_height - graph[p].output_shape[1])/2)
                    bottom = math.floor(
                        (in_height - graph[p].output_shape[1])/2)
                    left = math.floor((in_width - graph[p].output_shape[2])/2)
                    right = math.ceil((in_width - graph[p].output_shape[2])/2)
                    if (top, bottom, left, right) == (0, 0, 0, 0):
                        cell.conv_pad.append(None)
                    else:
                        cell.conv_pad.append(
                            keras.layers.ZeroPadding2D(
                                padding=((top, bottom), (left, right)),
                                name='conv_pad_{}'.format(cell_id))
                            )
            else:
                cell.prev = [-1]
                in_channels = input_shape[0]
                cell.conv_pad = [None]
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
            top = math.floor(padding_height/2)
            bottom = math.ceil(padding_height/2)
            left = math.floor(padding_width/2)
            right = math.ceil(padding_width/2)
            cell.conv_pad = [None]
        cell.in_channels = in_channels
        cell.conv = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=(filter_height, filter_width),
                strides=(stride_height, stride_width),
                padding="same",
                # activation='relu',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=keras.regularizers.l2(weight_decay),
                name="conv_{}".format(cell_id))
        cell.pool = keras.layers.MaxPooling2D(
                    pool_size=(pool_size, pool_size),
                    padding='same',
                    name='pool_{}'.format(cell_id))
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
        cell.pool_pad = None
        graph.append(cell)
        cell_id += 1
        prev_output_shape = cell.output_shape
    return graph


def empty_layer(x):
    return x


def compute_padding(input_size, kernel_size, stride):
    output_size = math.floor((input_size + stride - 1) / stride)
    padding = max(0, (output_size-1) * stride + kernel_size - input_size)
    return padding, output_size


if __name__ == "__main__":
    import time
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    arch_paras = [
        {'num_filters': 32, 'filter_height': 3, 'filter_width': 3,
         'stride_height': 1, 'stride_width': 1, 'pool_size': 1,
         'anchor_point': []},
        {'num_filters': 128, 'filter_height': 3, 'filter_width': 5,
         'stride_height': 1, 'stride_width': 1, 'pool_size': 1,
         'anchor_point': [1]},
        {'num_filters': 64, 'filter_height': 3, 'filter_width': 3,
         'stride_height': 1, 'stride_width': 1, 'pool_size': 2,
         'anchor_point': [0, 0]},
        {'num_filters': 96, 'filter_height': 5, 'filter_width': 5,
         'stride_height': 1, 'stride_width': 1, 'pool_size': 2,
         'anchor_point': [1, 0, 1]},
        {'num_filters': 128, 'filter_height': 3, 'filter_width': 7,
         'stride_height': 1, 'stride_width': 1, 'pool_size': 1,
         'anchor_point': [0, 0, 1, 1]},
        {'num_filters': 64, 'filter_height': 3, 'filter_width': 7,
         'stride_height': 1, 'stride_width': 2, 'pool_size': 1,
         'anchor_point': [0, 0, 0, 0, 1]}]

    quan_paras = []
    for l in range(len(arch_paras)):
        layer = {}
        layer['act_num_int_bits'] = 7
        layer['act_num_frac_bits'] = 7
        layer['weight_num_int_bits'] = 7
        layer['weight_num_frac_bits'] = 7
        quan_paras.append(layer)

    child_cnn = ChildCNN(arch_paras=arch_paras, input_shape=(32, 32, 3),
                         num_classes=10)
    start = time.time()
    child_cnn.train(validate=True, verbosity=1, epochs=40)
    end = time.time()
    print(f"Time: {end-start}")
    child_cnn.update_quantization(quan_paras)
    child_cnn.validate(verbosity=1, quantize=True)
