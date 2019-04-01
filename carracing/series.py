"""Uses a pretrained VAE to generate seires data to train an RNN"""
import argparse
import os
import shutil

from models.vae import get_vae
from tensorflow.keras.utils import HDF5Matrix
from tensorflow.keras.models import Model
import h5py
import numpy as np


TOTAL_NUM = 1000000
BATCH_SIZE = 1000


def scale(a):
    return a / 255


def main(args):
    data_path = args.data_path
    checkpoint_path = args.checkpoint_path
    temp_path = args.temp_path
    total_num = args.total_num
    batch_size = args.batch_size

    data_shape = (64, 64, 3)
    latent_dim = 32
    vae = get_vae(data_shape, latent_dim)
    vae.load_weights(checkpoint_path)
    encoder = Model(inputs=vae.input,
                    outputs=vae.get_layer('encoder').output)

    if not os.path.exists(os.path.split(temp_path)[0]):
        os.makedirs(os.path.split(temp_path)[0])

    with h5py.File(temp_path, 'a') as output_file:
        x = HDF5Matrix(data_path, 'obs', normalizer=scale)
        if total_num == 0: total_num = len(x)
        print('Generating', total_num, 'data points.')
        
        zs = output_file.create_dataset('z',
                                        shape=(total_num, latent_dim),
                                        dtype=np.float32)

        for i in range(0, total_num, batch_size):
            print(i, '/', total_num, end='\r')
            zs[i:i+BATCH_SIZE] = encoder.predict(x[i:i+batch_size])[-1]

        # Close HDF5 file in x
        x.refs[data_path].close()

    with h5py.File(data_path, 'a') as output_file:
        with h5py.File(temp_path, 'r') as input_file:
                zs_out = output_file.create_dataset('z',
                                                    data=input_file.get('z'))

    shutil.rmtree('temp/')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate rnn trianing data.')
    parser.add_argument('--data_path', '-d', default='data/train.h5')
    parser.add_argument('--checkpoint_path', '-c', default='checkpoints/vae.h5')
    parser.add_argument('--temp_path', default='temp/temp.h5')
    parser.add_argument('--total_num', '-t', type=int, default=0)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    main(parser.parse_args())
