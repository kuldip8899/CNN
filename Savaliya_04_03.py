# Savaliya, Kuldip
# 1001_832_000
# 2022_11_13
# Assignment_04_03

import numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

from cnn import CNN


def test_train():
    
    test_images = 280
    classes = 10
    batch_size=32
    epochs = 10
    (train_X, train_y), (test_x, test_y) = cifar10.load_data()
    train_X = (train_X[0:test_images, :]).astype('float64') / 255
    test_x = (test_x[0:test_images, :]).astype('float64') / 255
    train_y = keras.utils.to_categorical(train_y[0:test_images, :], classes)
    test_y = keras.utils.to_categorical(test_y[0:test_images, :], classes)

    model = keras.Sequential()
    model.add(Conv2D(64, (5, 5), padding = 'valid', activation = 'linear', input_shape = train_X.shape[1:]))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(32, (5, 5), padding = 'valid', activation = 'linear'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(50, activation = 'linear'))
    model.add(Dense(classes, activation = 'softmax'))
    optimal_parameters = keras.optimizers.RMSprop(learning_rate = 0.001)
    model.compile(optimizer=optimal_parameters, loss="hinge", metrics=['accuracy'])
    model_fit = model.fit(train_X, train_y, batch_size = batch_size, epochs = epochs)
    
    cnn = CNN()
    cnn.add_input_layer(shape=train_X.shape[1:], name="input")
    cnn.append_conv2d_layer(num_of_filters=64, kernel_size=(3,3),padding="same", activation='linear', strides = 2,  name="conv1")
    cnn.append_maxpooling2d_layer(pool_size=2, padding="same",  strides = 1, name="pool1")
    cnn.append_conv2d_layer(num_of_filters=32, kernel_size=3, activation='linear', name="conv2")
    cnn.append_maxpooling2d_layer(pool_size=2, padding="same",  strides = 2, name="pool2")
    cnn.append_flatten_layer(name="flat1")
    cnn.append_dense_layer(num_nodes=50,activation="linear",name="dense1")
    cnn.append_dense_layer(num_nodes=classes,activation="softmax",name="dense2")
    cnn.set_optimizer("adagrad", 0.001)
    cnn.set_metric("accuracy")
    cnn.set_loss_function("hinge")
    loss = cnn.train(train_X, train_y, batch_size=batch_size, num_epochs=epochs)

    assert np.allclose(model_fit.history['loss'], loss, rtol=1e-1, atol=1e-1 * 6)



def test_evaluate():

    test_images = 280
    classes = 10
    batch_size=32
    epochs = 10
    (train_X, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
    train_X = (train_X[0:test_images, :]).astype('float64') / 255
    test_x = (test_x[0:test_images, :]).astype('float64') / 255
    train_y = keras.utils.to_categorical(train_y[0:test_images, :], classes)
    test_y = keras.utils.to_categorical(test_y[0:test_images, :], classes)

    model = keras.Sequential()
    model.add(Conv2D(64, (5, 5), padding = 'valid', activation = 'linear', input_shape = train_X.shape[1:]))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(32, (3, 3), padding = 'valid', activation = 'linear'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(50, activation = 'linear'))
    model.add(Dense(classes, activation = 'softmax'))
    optimal_parameters = keras.optimizers.RMSprop(learning_rate = 0.001)
    model.compile(optimizer=optimal_parameters, loss="hinge", metrics=['accuracy'])
    model.fit(train_X, train_y, batch_size = batch_size, epochs = epochs)
    accuracy = model.evaluate(test_x, test_y)

    cnn = CNN()
    cnn.add_input_layer(shape=train_X.shape[1:], name="input")
    cnn.append_conv2d_layer(num_of_filters=64, kernel_size=(3,3),padding="same", activation='linear', strides = 1,  name="conv1")
    cnn.append_maxpooling2d_layer(pool_size=2, padding="same",  strides = 1, name="pool1")
    cnn.append_conv2d_layer(num_of_filters=32, kernel_size=3, activation='linear', name="conv2")
    cnn.append_maxpooling2d_layer(pool_size=2, padding="same",  strides = 1, name="pool2")
    cnn.append_flatten_layer(name="flat1")
    cnn.append_dense_layer(num_nodes=50,activation="linear",name="dense1")
    cnn.append_dense_layer(num_nodes=classes,activation="softmax",name="dense2")
    cnn.set_optimizer("adagrad", 0.001)
    cnn.set_metric("accuracy")
    cnn.set_loss_function("hinge")
    cnn.train(train_X, train_y, batch_size=batch_size, num_epochs=epochs)
    accuracy_cnn = cnn.evaluate(test_x, test_y)

    assert np.allclose(accuracy, accuracy_cnn, rtol=1e-1, atol=1e-1 * 6)
