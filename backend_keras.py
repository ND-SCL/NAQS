import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class IdentityLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(IdentityLayer, self).__init__(**kwargs)

    def call(self, input):
        return input


class QuanLayer(tf.keras.layers.Layer):
    def __init__(self, num_int_bits, num_frac_bits, signed=False, **kwargs):
        self.num_int_bits = num_int_bits
        self.num_frac_bits = num_frac_bits
        self.signed = signed
        super(QuanLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.precision = self.add_variable("precision", shape=[])
        self.precision = 1 / 2 ** (self.num_frac_bits)

    def call(self, input):
        if self.signed:
            bound = 2 ** (self.num_int_bits - 1)
            return tf.clip_by_value(
                tf.round(input / self.precision) * self.precision,
                -bound,
                bound - self.precision
                )
        else:
            bound = 2 ** (self.num_int_bits)
            return tf.clip_by_value(
                tf.round(input / self.precision) * self.precision,
                0,
                bound - self.precision
                )


def get_cifar10(percentage=1):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    num_samples = int(len(x_train) * percentage)
    x_train = x_train[:num_samples]
    y_train = y_train[:num_samples]
    num_samples = int(len(x_test) * percentage)
    x_test = x_test[:num_samples]
    y_test = y_test[:num_samples]
    x_train.astype('float32')
    x_train = x_train / 255
    x_test.astype('float32')
    x_test = x_test / 255
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    train_images, train_labels = x_train, y_train
    val_images, val_labels = x_test, y_test
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_images = normalize(train_images, mean, std)
    val_images = normalize(val_images, mean, std)
    return (train_images, train_labels), (val_images, val_labels)


datagen_augment = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen_plain = ImageDataGenerator()


def fit(model, validate=False, quantize=False, verbosity=0, epochs=40):
    history = trainer(
        model,
        epochs=epochs,
        validate=validate,
        verbosity=verbosity)
    if validate is True:
        if quantize is True:
            return validator(model)
        else:
            loss = history.history['val_loss'][-5:],
            acc = history.history['val_acc'][-5:]
            return np.mean(loss), np.mean(acc)
    else:
        loss = history.history['loss'][-5:],
        acc = history.history['acc'][-5:]
        return np.mean(loss), np.mean(acc)


def trainer(model, epochs=40, batch_size=128, validate=False, verbosity=0):
    (train_images, train_labels), (valid_images, valid_labels) = get_cifar10()
    optimizer = get_optimizer('Adam')
    datagen_plain.fit(train_images)

    def lr_scheduler(epoch):
        lr = 1e-3
        if epoch > 15:
            lr = 3e-4
        if epoch > 25:
            lr = 8e-5
        return lr
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )
    history = model.fit_generator(
        datagen_plain.flow(train_images, train_labels, batch_size=batch_size),
        epochs=epochs,
        validation_data=(valid_images, valid_labels) if validate else None,
        callbacks=[reduce_lr],
        verbose=verbosity)
    return history


def validator(model, verbosity=0):
    (train_images, train_labels), (valid_images, valid_labels) = get_cifar10()
    optimizer = get_optimizer('SGD')
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    loss, acc = model.evaluate(valid_images, valid_labels, verbose=verbosity)
    return loss, acc


def get_info(dataset='CIFAR10'):
    return (32, 32, 3), 10


def normalize(batch, mean, std):
    for i in range(len(mean)):
        print(i)
        if std[i] == 0:
            std[i] = 1e-8
        batch[:, :, :, i] = (batch[:, :, :, i] - mean[i]) / std[i]
    return batch


def get_optimizer(name='SGD'):
    adam_optim = keras.optimizers.Adam(
        lr=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        decay=0,
        amsgrad=False
        )
    sgd_optim = keras.optimizers.SGD(
        lr=0.01,
        momentum=0.9,
        decay=5e-4,
        nesterov=True)
    if name == 'SGD':
        return sgd_optim
    else:
        return adam_optim
