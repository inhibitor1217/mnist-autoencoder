import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

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
    x = _input = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    model = Model(_input, x, name=model_name)
    model.summary()

    return model

def Decoder(model_name):
    x = _input = Input(shape=(4, 4, 8))
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(16, (3, 3))(x)
    x = LeakyReLU()(x)
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

    # train_x_noise = train_x + .5 * np.random.normal(0., 1., size=train_x.shape)
    # test_x_noise  = test_x  + .5 * np.random.normal(0., 1., size=test_x .shape)

    encoder = Encoder('')
    decoder = Decoder('')
    autoencoder = AE(encoder, decoder, '')

    parallel_model = multi_gpu_model(autoencoder, gpus=2)
    parallel_model.compile(optimizer=Adam(lr=1e-2), loss='binary_crossentropy')

    parallel_model.fit(train_x, train_x, epochs=5, shuffle=True, batch_size=128,
                    # validation_data=(test_x, test_x),
                    callbacks=[
                        TensorBoard(log_dir='./tmp/autoencoder'),
                        # ModelCheckpoint('./checkpoint/model-{epoch:02d}.h5', save_freq='epoch')
                    ])

    t = test_x[:50]

    out = parallel_model.predict(t)

    out_img = np.zeros(shape=(280, 280))
    for i in range(10):
        for j in range(5):
            out_img[i*28:(i+1)*28, 2*j*28:(2*j+1)*28] = t[5*i+j].reshape(28, 28)
            out_img[i*28:(i+1)*28, (2*j+1)*28:(2*j+2)*28] = out[5*i+j].reshape(28, 28)

    plt.imshow(out_img)
    plt.gray()

    # autoencoder.load_weights('./checkpoint/model-25.h5')
    #
    # decoded_imgs = autoencoder.predict(test_x_noise, batch_size=128)
    #
    # n = 10
    # plt.figure(figsize=(20, 4))
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(2, n, i+1)
    #     plt.imshow(test_x_noise[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #
    #     # display reconstruction
    #     ax = plt.subplot(2, n, i+1 + n)
    #     plt.imshow(decoded_imgs[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    # plt.show()

if __name__ == '__main__':
    main()