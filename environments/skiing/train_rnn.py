"""Trains the rnn on data created using `series.py`."""
import argparse
import os
from random import shuffle

from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

import config

import sys
sys.path.append('../../')
from models.rnn import get_rnn


def main(args):
    data_dir = args.data_dir
    epochs = args.epochs
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path

    latent_dim = config.latent_dim
    lstm_dim = config.lstm_dim
    lstm_num_mixtures = config.lstm_num_mixtures

    raw_data = np.load(os.path.join(data_dir, 'series.npz'))
    zs = raw_data['z']
    actions = raw_data['action']

    # Combine encoder output and action
    x_train = np.concatenate([zs, actions], axis=-1)
    # Offset x and y
    x_train = x_train[:,:-1]
    y_train = zs[:,1:]

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='train_loss')
    rnn, _ = get_rnn(x_train.shape[1:],
                     lstm_dim=lstm_dim,
                     output_sequence_width=latent_dim,
                     num_mixtures=lstm_num_mixtures,
                     train=True)

    rnn.fit(x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[checkpoint])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train the rnn.')
    parser.add_argument('--data_dir', '-d', default='data/',
			            help='The path to the training data directory.')
    parser.add_argument('--epochs', '-e', type=int, default=40,
			            help='The number of epochs to train for.')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
			            help='The batch size to use for training.')
    parser.add_argument('--checkpoint_path', default='checkpoints/rnn.h5',
			            help='The path to save the checkpoint at.')
    main(parser.parse_args())
