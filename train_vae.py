import os
import argparse

import numpy as np

from vae import get_vae

def main():
    parser = argparse.ArgumentParser(description='Train the vae.')
    parser.add_argument('data_dir', help='The directory which the data is placed in.')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='The number of epochs to train for.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='The batch size to use for training.')
    args = parser.parse_args()
    data_dir = args.data_dir
    epcohs = args.epochs
    batch_size = args.batch_size

    x = np.load(os.path.join(data_dir, 'pong.npy'))

    vae = get_vae(x.shape[1:], 32)
    vae.compile(optimizer='adam')
    vae.fit(x,
            epochs=epochs,
            batch_size=batch_size)
    vae.save_weights('vae.h5')

if __name__=='__main__':
    main()
