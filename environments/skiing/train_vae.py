"""Trains the vae on data created using `extract.py`."""
import argparse
import os
from random import shuffle

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

import config

import sys
sys.path.append('../../')
from models.vae import get_vae
from data_utils import generate_observation_data


def main(args):
    data_dir = args.data_dir
    epochs = args.epochs
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path
    num_episodes = args.num_episodes
    val_episodes = args.val_episodes
    num_frames = args.num_frames
    load = args.load

    data_shape = config.input_shape
    latent_dim = config.latent_dim

    vae = get_vae(data_shape, latent_dim,
                  filters=[16, 32, 64, 128, 256],
                  kernels=[4, 4, 4, 4, 4],
                  strides=[2, 2, 2, 2, 2],
                  deconv_filters=[256, 128, 64, 32, 16, 3],
                  deconv_kernels=[2, 5, 4, 4, 5, 4],
                  deconv_strides=[2, 2, 2, 2, 2, 2],
                  train=True)

    if load:
        vae.load_weights(checkpoint_path)

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='train_loss')

    vae.fit_generator(generate_observation_data(data_dir,
                                                batch_size,
                                                num_episodes,
                                                num_frames),
                        steps_per_epoch=(num_episodes * num_frames / batch_size),
                        epochs=epochs,
                        workers=28,
                        validation_data=generate_observation_data(data_dir,
                                                                  batch_size,
                                                                  val_episodes,
                                                                  num_frames,
                                                                  offset=num_episodes),
                        validation_steps=(val_episodes * num_frames / batch_size),
                        callbacks=[checkpoint])

    vae.save_weights('checkpoints/vae_final.h5')


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
    parser.add_argument('--val_episodes', type=int, default=1000,
                        help='The number of episodes to use for validation.')
    parser.add_argument('--num_frames', type=int, default=1000,
                        help='The number of frames per episode.')
    parser.add_argument('--load', action='store_true',
                        help=('Loads the network from the checkpoint path'
                              'before beginning training.'))
    main(parser.parse_args())
