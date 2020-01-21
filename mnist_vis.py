import matplotlib.pyplot as plt
import numpy as np
from mnist_ae import load_mnist_data, Encoder, Decoder, AE

def main():
    encoder = Encoder('encoder')
    decoder = Decoder('decoder')
    autoencoder = AE(encoder, decoder, 'auto_encoder')

    autoencoder.load_weights('./checkpoint/model-15.h5')

    _, (x, y) = load_mnist_data()

    x_noisy = x + .5 * np.random.normal(loc=0.0, scale=1.0, size=x.shape)

    n = 10
    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        plt.imshow(x_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    x_filtered = autoencoder.predict(x_noisy, batch_size=128)

    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(x_filtered[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    main()