import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Lambda, LeakyReLU
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def Encoder():
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(2,), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_sigma) * epsilon

    x = _input = Input(shape=(784,))

    x = Dense(512)(x)
    x = LeakyReLU()(x)

    z_mean = Dense(2, name='features')(x)
    z_log_sigma = Dense(2)(x)
    z = Lambda(sampling, output_shape=(2,))([z_mean, z_log_sigma])

    model = Model(inputs=_input, outputs=[z_mean, z_log_sigma, z])
    return model

def Decoder():
    x = _input = Input(shape=(2,))

    x = Dense(512)(x)
    x = LeakyReLU()(x)

    x = Dense(784, activation='sigmoid')(x)

    model = Model(inputs=_input, outputs=x)
    return model

def VAE(encoder, decoder):

    _input = Input(shape=(784,))

    [z_mean, z_log_sigma, z] = encoder(_input)

    kl_divergence_loss = -.5 * K.sum(1. + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)

    x = decoder(z)

    model = Model(inputs=_input, outputs=[x, kl_divergence_loss])
    return model

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train / 255.).reshape((x_train.shape[0], 784))
    x_test  = (x_test  / 255.).reshape((x_test.shape[0],  784))
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    encoder = Encoder()
    decoder = Decoder()
    vae     = VAE(encoder, decoder)

    vae.compile(optimizer=Adam(lr=1e-3), loss=['binary_crossentropy', 'mae'], loss_weights=[784, 1])

    vae.fit(x_train, [x_train, np.zeros((x_train.shape[0],))], shuffle=True, epochs=50, batch_size=1024)

    [t, _, _] = encoder.predict(x_test)
    plt.scatter(t[:, 0], t[:, 1], c=y_test)