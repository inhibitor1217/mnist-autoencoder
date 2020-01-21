import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import matplotlib.pyplot as plt

def normalize(img):
    return img / 255.

def denormalize(img):
    return img * 255.

def load_mnist_data():
    print('Loading MNIST dataset ...')

    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # Normalize image
    train_x = normalize(train_x)
    test_x  = normalize(test_x)

    # Fit image to (28, 28, 1)
    train_x = np.expand_dims(train_x, axis=-1)
    test_x  = np.expand_dims(test_x,  axis=-1)

    return (train_x, train_y), (test_x, test_y)

def Encoder(model_name):
    _input = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(_input)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    model = Model(_input, x, name=model_name)
    model.summary()

    return model

def Decoder(model_name):
    _input = Input(shape=(4, 4, 8))
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(_input, x, name=model_name)
    model.summary()

    return model

def AE(encoder, decoder, model_name):
    _input = Input(shape=(28, 28, 1))
    encoded = encoder(_input)
    decoded = decoder(encoded)

    model = Model(_input, decoded, name=model_name)

    return model

def main():
    (train_x, _), (test_x, _) = load_mnist_data()

    train_x_noise = train_x + .5 * np.random.normal(0., 1., size=train_x.shape)
    test_x_noise  = test_x  + .5 * np.random.normal(0., 1., size=test_x .shape)

    encoder = Encoder('')
    decoder = Decoder('')
    autoencoder = AE(encoder, decoder, '')

    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    #
    # autoencoder.fit(train_x_noise, train_x, epochs=25, batch_size=128, shuffle=True,
    #                 validation_data=(test_x_noise, test_x),
    #                 callbacks=[
    #                     TensorBoard(log_dir='./tmp/autoencoder'),
    #                     ModelCheckpoint('./checkpoint/model-{epoch:02d}.h5', save_freq='epoch')
    #                 ])

    autoencoder.load_weights('./checkpoint/model-25.h5')

    decoded_imgs = autoencoder.predict(test_x_noise, batch_size=128)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(test_x_noise[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i+1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    main()