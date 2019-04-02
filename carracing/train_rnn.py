"""Trains the rnn on data created using `extract.py`."""
import argparse

from models.rnn import get_rnn
from tensorflow.keras.utils import HDF5Matrix
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
# Debugging
import tensorflow as tf
import tensorflow.keras.backend as K


def main(args):
    data_path = args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path
    end = args.data_length

    zs = HDF5Matrix(data_path, 'z', end=end)
    actions = HDF5Matrix(data_path, 'action', end=end)

    latent_dim = zs.shape[-1]
    action_dim = actions.shape[-1]
    data_shape = (None, latent_dim + action_dim)
    x_train = np.column_stack([zs, actions])[:-1]
    x_train = np.reshape(x_train, (-1, 1, latent_dim + action_dim))
    y_train = zs[1:]

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='train_loss')

    rnn = get_rnn(data_shape)
    rnn.compile(loss=rnn.loss, optimizer='adam')
    rnn.fit(x_train, y_train,
	        epochs=epochs,
	        batch_size=batch_size,
	        shuffle='batch',
            callbacks=[checkpoint])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train the rnn.')
    parser.add_argument('--data_path', '-d', default='data/train.h5',
			help='The path to the training data.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
			help='The number of epochs to train for.')
    parser.add_argument('--batch_size', '-b', type=int, default=128,
			help='The batch size to use for training.')
    parser.add_argument('--data_length', '-', default=1000000,
                        help='The length of input data to use.')
    parser.add_argument('--checkpoint_path', default='checkpoints/rnn.h5',
			help='The path to save the checkpoint at.')
    main(parser.parse_args())
