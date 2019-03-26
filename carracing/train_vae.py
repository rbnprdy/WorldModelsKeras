import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from vae import get_vae

# preprocessing used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float)

def main():
    parser = argparse.ArgumentParser(description='Train the vae.')
    parser.add_argument('data_dir', help='The directory which the data is placed in.')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='The number of epochs to train for.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='The batch size to use for training.')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualizes the output of trained autoencoder instead of training.')
    args = parser.parse_args()
    visualize = args.visualize
    data_dir = args.data_dir
    epochs = args.epochs
    batch_size = args.batch_size

    x = np.load(os.path.join(data_dir, 'pong.npy'))
    x_prepro = np.zeros((x.shape[0], 80, 80))
    for i, im in enumerate(x):
        x_prepro[i] = prepro(im)
    x_prepro = x_prepro[:,:,:, np.newaxis]
    vae = get_vae(x_prepro.shape[1:], 32)

    if visualize:
        vae.load_weights('vae.h5')
        vae.compile(optimizer='rmsprop')
        predictions = vae.predict(x_prepro[50:55])
        figs, axes = plt.subplots(5, 2)
        for im, prediction, ax in zip(x_prepro[50:55], predictions, axes):
            print(im[:-1].shape)
            ax[0].imshow(im[:,:,0])
            ax[0].set_xlabel('Original')
            ax[1].imshow(prediction[:,:,0])
            ax[1].set_xlabel('Reconstructed')
        plt.show() 
    else:
        vae.compile(optimizer='rmsprop')
        vae.fit(x_prepro,
                epochs=epochs,
                batch_size=batch_size)
        vae.save_weights('vae.h5')

if __name__=='__main__':
    main()
