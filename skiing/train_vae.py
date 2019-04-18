"""Trains the vae on data created using `extract.py`."""
import argparse
import os

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

from models.vae import get_vae


def create_dataset(data_dir, filelist, N=10000, M=1000):
    data = np.zeros((M*N, 64, 64, 3), dtype=np.uint8)
    idx = 0
    for i in range(N):
        filename = filelist[i]
        raw_data = np.load(os.path.join(data_dir, filename))['obs']
        l = len(raw_data)
        if (idx+l) > (M*N):
            data = data[0:idx]
            print('premature break')
            break
        data[idx:idx+l] = raw_data
        idx += l
        if ((i+1) % 100 == 0):
            print("loading file", i+1)
    return data


def main(args):
        data_dir = args.data_dir
        epochs = args.epochs
        batch_size = args.batch_size
        checkpoint_path = args.checkpoint_path
        num_episodes = args.num_episodes
        num_frames = args.num_frames

        filelist = os.listdir(data_dir)
        filelist.sort()
        filelist = filelist[0:10000]

        dataset = create_dataset(data_dir, filelist, N=num_episodes, M=num_frames)

        data_shape = (144, 144, 3)
        vae = get_vae(data_shape, 32, scale_input=True,
                      filters=[16, 32, 64, 128, 256],
                      kernels=[4, 4, 4, 4, 4],
                      strides=[2, 2, 2, 2, 2],
                      deconv_filters=[256, 128, 64, 32, 16, 3],
                      deconv_kernels=[2, 5, 4, 4, 5, 4],
                      deconv_strides=[2, 2, 2, 2, 2, 2])

        checkpoint = ModelCheckpoint(checkpoint_path, monitor='train_loss')

        vae.compile(optimizer='adam')
        vae.fit(dataset,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[checkpoint])


if __name__=='__main__':
        parser = argparse.ArgumentParser(description='Train the vae.')
        parser.add_argument('--data_dir', '-d', default='data/',
                                                help=('The path to the folder containing the training'
                                                      ' data.'))
        parser.add_argument('--epochs', '-e', type=int, default=10,
                                                help='The number of epochs to train for.')
        parser.add_argument('--batch_size', '-b', type=int, default=128,
                                                help='The batch size to use for training.')
        parser.add_argument('--checkpoint_path', default='checkpoints/vae.h5',
                                                help='The path to save the checkpoint at.')
        parser.add_argument('--num_episodes', type=int, default=10000,
                                                help='The number of episodes to use for training.')
        parser.add_argument('--num_frames', type=int, default=1000,
                                                help='The number of frames per episode.')
        main(parser.parse_args())
