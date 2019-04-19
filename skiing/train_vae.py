"""Trains the vae on data created using `extract.py`."""
import argparse
import os
from random import shuffle

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

from models.vae import get_vae


IMAGE_SIZE = (144, 144, 3)


def generate_data(data_dir, batch_size, num_episodes, num_frames):
    assert(num_frames % batch_size == 0,
           'num_frames must be divisible by batch_size because I am lazy')
    
    filelist = os.listdir(data_dir)
    filelist.sort()
    filelist = filelist[0:num_episodes]
    shuffle(filelist)
    
    file_num = 0
    image_num = 0    
    
    curr_file = np.load(os.path.join(data_dir, filelist[file_num]))['obs'].astype(np.float) / 255.
    while True:
        batch = curr_file[image_num:image_num+batch_size]
        np.random.shuffle(batch)

        image_num += batch_size
        if image_num == num_frames:
            image_num = 0
            file_num += 1
            if file_num == len(filelist):
                file_num = 0
                shuffle(filelist)
            curr_file = np.load(os.path.join(data_dir, filelist[file_num]))

        yield (batch)


def main(args):
    data_dir = args.data_dir
    epochs = args.epochs
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path
    num_episodes = args.num_episodes
    num_frames = args.num_frames

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
    vae.fit_generator(generate_data(data_dir,
                                    batch_size,
                                    num_episodes,
                                    num_frames),
                      steps_per_epoch=(num_episodes * num_frames / batch_size),
                      epochs=epochs,
                      workers=28,
                      use_multiprocessing=True,
                      callbacks=[checkpoint])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train the vae.')
    parser.add_argument('--data_dir', '-d', default='data/',
                        help=('The path to the folder containing the training'
                                ' data.'))
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='The number of epochs to train for.')
    parser.add_argument('--batch_size', '-b', type=int, default=100,
                        help='The batch size to use for training.')
    parser.add_argument('--checkpoint_path', default='checkpoints/vae.h5',
                        help='The path to save the checkpoint at.')
    parser.add_argument('--num_episodes', type=int, default=10000,
                        help='The number of episodes to use for training.')
    parser.add_argument('--num_frames', type=int, default=1000,
                        help='The number of frames per episode.')
    main(parser.parse_args())
