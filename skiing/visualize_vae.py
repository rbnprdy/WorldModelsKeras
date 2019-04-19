"""Generates a figure of the VAE recreation on random samples"""
import argparse
import os

from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

from models.vae import get_vae


def main(args):
    data_dir = args.data_dir
    checkpoint_path = args.checkpoint_path
    output_file = args.output_file
    num_images = args.num_images

    data_shape = (144, 144, 3)
    latent_dim = 32

    vae = get_vae(data_shape, latent_dim,
                  filters=[16, 32, 64, 128, 256],
                  kernels=[4, 4, 4, 4, 4],
                  strides=[2, 2, 2, 2, 2],
                  deconv_filters=[256, 128, 64, 32, 16, 3],
                  deconv_kernels=[2, 5, 4, 4, 5, 4],
                  deconv_strides=[2, 2, 2, 2, 2, 2])
    vae.load_weights(checkpoint_path)

    filelist = os.listdir(data_dir)
    filelist.sort()
    filename = filelist[0]
    obs = np.load(os.path.join(data_dir, filename))['obs'].astype(np.float) / 255.
    np.random.shuffle(obs)
    obs = obs[:num_images]
    predictions = vae.predict(obs)

    fig, axes = plt.subplots(num_images, 2)
    for ax, im, gen in zip(axes, obs, predictions):
        ax[0].imshow(im, interpolation='nearest')
        ax[0].set_title('Original Image')
        ax[1].imshow(gen, interpolation='nearest')
        ax[1].set_title('Autoencoder Recreation')
    plt.savefig(output_file)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Visualize the VAE.')
    parser.add_argument('--data_dir', '-d', default='data/')
    parser.add_argument('--checkpoint_path', '-c', default='checkpoints/vae.h5')
    parser.add_argument('--output_file', '-o', default='vae.png')
    parser.add_argument('--num_images', '-n', type=int, default=5)
    main(parser.parse_args())
