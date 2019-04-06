"""Uses a pretrained VAE to generate seires data to train an RNN"""
import argparse
import os
import shutil

from models.vae import get_vae
from tensorflow.keras.utils import HDF5Matrix
from tensorflow.keras.models import Model
import h5py
import numpy as np


def scale(a):
    return a / 255


def main(args):
    data_path = args.data_path
    checkpoint_path = args.checkpoint_path
    output_path = args.output_path
    total_num = args.total_num
    batch_size = args.batch_size

    data_shape = (64, 64, 3)
    latent_dim = 32
    action_dim = 3
    vae = get_vae(data_shape, latent_dim)
    vae.load_weights(checkpoint_path)
    encoder = Model(inputs=vae.input,
                    outputs=vae.get_layer('encoder').output)

    if not os.path.exists(os.path.split(output_path)[0]):
        os.makedirs(os.path.split(output_path)[0])

    with h5py.File(output_path, 'a') as output_file:
        x = HDF5Matrix(data_path, 'obs', normalizer=scale)
        action_in = HDF5Matrix(data_path, 'action')
        if total_num == 0: total_num = len(x)
        print('Generating', total_num, 'data points.')

        zs = output_file.create_dataset('z',
                                        shape=(total_num // batch_size,
                                               batch_size,
                                               latent_dim),
                                        dtype=np.float32)

        # FIXME: I should have just saved everything as a series
        # in the first place
        action_out = output_file.create_dataset('action',
                                                shape=(total_num // batch_size,
                                                       batch_size,
                                                       action_dim),
                                                dtype=np.float16)

        for i in range(total_num // batch_size):
            print(i, '/', total_num // batch_size, end='\r')
            zs[i] = encoder.predict(x[i*batch_size:(i+1)*batch_size])[-1]
            action_out[i] = action_in[i*batch_size:(i+1)*batch_size]


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate rnn trianing data.')
    parser.add_argument('--data_path', '-d', default='data/train.h5')
    parser.add_argument('--checkpoint_path', '-c', default='checkpoints/vae.h5')
    parser.add_argument('--output_path', default='data/series.h5')
    parser.add_argument('--total_num', '-t', type=int, default=0)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    main(parser.parse_args())
