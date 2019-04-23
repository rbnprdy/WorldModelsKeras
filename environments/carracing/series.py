"""Uses the trained VAE to generate seires data to train the RNN"""
import argparse
import os

from tensorflow.keras.models import Model
import numpy as np

import config

import sys
sys.path.append('../../')
from models.vae import get_vae


def main(args):
    data_dir = args.data_dir
    checkpoint_path = args.checkpoint_path
    total_num = args.total_num
    batch_size = args.batch_size

    data_shape = config.data_shape
    latent_dim = config.latent_dim
    action_dim = config.action_dim

    vae = get_vae(data_shape, latent_dim)
    vae.load_weights(checkpoint_path)
    encoder = Model(inputs=vae.input,
                    outputs=vae.get_layer('encoder').output)

    filelist = os.listdir(data_dir)
    filelist.sort()
    filelist = filelist[0:total_num]

    action_dataset = []
    mu_dataset = []
    sigma_dataset = []
    z_dataset = []
    for filename in filelist:
        raw_data = np.load(os.path.join(data_dir, filename))
        action_dataset.append(raw_data['action'])
        mu, sigma, z = encoder.predict(raw_data['obs'].astype(np.float) / 255.)
        mu_dataset.append(mu)
        sigma_dataset.append(sigma)
        z_dataset.append(z)
        
    action_dataset = np.array(action_dataset)
    mu_dataset = np.array(mu_dataset)
    sigma_dataset = np.array(sigma_dataset)
    z_dataset = np.array(z_dataset)

    np.savez_compressed(os.path.join(data_dir, "series.npz"),
                        action=action_dataset,
                        mu=mu_dataset,
                        sigma=sigma_dataset,
                        z=z_dataset)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate rnn trianing data.')
    parser.add_argument('--data_dir', '-d', default='data/')
    parser.add_argument('--checkpoint_path', '-c', default='checkpoints/vae.h5')
    parser.add_argument('--total_num', '-t', type=int, default=11000)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    main(parser.parse_args())