import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train.astype('float32')
x_train = x_train / 255
x_test.astype('float32')
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

split_index = int(0.9 * len(x_train))
train_images, valid_images = x_train[:split_index], x_train[split_index:]
train_labels, valid_labels = y_train[:split_index], y_train[split_index:]
test_images = x_test
test_labels = y_test

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms


# batch_size = 128
# data_utilization = 1



# transform = transforms.Compose(
#     [transforms.ToTensor()])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# trainset, validset = torch.utils.data.random_split(trainset, [45000, 5000])

# # _, validset = torch.utils.data.random_split(trainset, [45000, 5000])
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=45000, shuffle=False, num_workers=2)
# validloader = torch.utils.data.DataLoader(validset, batch_size=5000, shuffle=False, num_workers=2)

# trainiter = iter(trainloader)
# train_images, train_labels = trainiter.next()
# train_images = np.transpose(train_images.numpy(), (0, 2, 3, 1))
# train_labels = train_labels.numpy()
# train_labels = keras.utils.to_categorical(train_labels, num_classes=10)

# validiter = iter(validloader)
# valid_images, valid_labels = validiter.next()
# valid_images = np.transpose(valid_images.numpy(), (0, 2, 3, 1))
# valid_labels = valid_labels.numpy()
# valid_labels = keras.utils.to_categorical(valid_labels, num_classes=10)



# print(train_images.shape)
# print(train_labels.shape)




model = keras.Sequential()
model.add(keras.layers.Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax'))

datagen = ImageDataGenerator()
datagen.fit(train_images)

optimizer = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# input_ = np.ones((1, 32, 32, 3))
# model.predict(input_)
# for layer in model.layers:
#     print(layer.input)

history = model.fit_generator(datagen.flow(train_images, train_labels, batch_size=128), 
	steps_per_epoch = int(45000 / 128),
	epochs = 150, validation_data = (valid_images, valid_labels))





