import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Cropping2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import matplotlib.pyplot as plt

def normalize(img):
    return img / 127.5 - 1.

def denormalize(img):
    return (img + 1.) + 127.5

def load_mnist_data():
    print('Loading MNIST dataset ...')

    (train_x, _), (test_x, _) = mnist.load_data()

    # Normalize image
    train_x = normalize(train_x)
    test_x  = normalize(test_x)

    # Fit image to (28, 28, 1)
    train_x = np.expand_dims(train_x, axis=-1)
    test_x  = np.expand_dims(test_x,  axis=-1)

    return train_x, test_x

def Encoder(model_name):
    _input = Input(shape=(28, 28, 1), name=f'{model_name}_input')
    x = Conv2D(filters=16, kernel_size=3, strides=2, padding='same',
                activation='relu', name=f'{model_name}_conv1')(_input)
    x = Conv2D(filters=8, kernel_size=3, strides=2, padding='same',
                activation='relu', name=f'{model_name}_conv2')(x)
    x = Conv2D(filters=8, kernel_size=3, strides=2, padding='same',
                activation='relu', name=f'{model_name}_conv3')(x)

    model = Model(_input, x, name=model_name)
    model.summary()

    return model

def Decoder(model_name):
    _input = Input(shape=(4, 4, 8), name=f'{model_name}_input')
    x = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same',
                        activation='relu', name=f'{model_name}_deconv1')(_input)
    x = Conv2DTranspose(filters=8,  kernel_size=3, strides=2, padding='same',
                        activation='relu', name=f'{model_name}_deconv2')(x)
    x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same',
                        activation='relu', name=f'{model_name}_deconv3')(x)
    x = Conv2D(filters=1, kernel_size=3, strides=1, padding='same',
               activation='tanh', name=f'{model_name}_output')(x)
    x = Cropping2D(cropping=2, name=f'{model_name}_cropping')(x)

    model = Model(_input, x, name=model_name)
    model.summary()

    return model

def AE(encoder, decoder, model_name):
    _input = Input(shape=(28, 28, 1), name=f'{model_name}_autoencoder')

    encoded = encoder(_input)
    decoded = decoder(encoded)

    model = Model(_input, decoded, name=model_name)

    return model

def main():
    encoder = Encoder('encoder')
    decoder = Decoder('decoder')

    autoencoder = AE(encoder, decoder, 'auto_encoder')

    optimizer = Adam(lr=1e-3)
    autoencoder.compile(optimizer=optimizer, loss='mae')

    train_x, test_x = load_mnist_data()

    autoencoder.fit(train_x, train_x, epochs=15, batch_size=128, shuffle=True,
                    validation_data=(test_x, test_x),
                    callbacks=[
                        TensorBoard(log_dir='./tmp/auto_encoder'),
                        ModelCheckpoint('./checkpoint/model-{epoch:02d}.h5', save_freq=5)
                    ])

if __name__ == '__main__':
    main()