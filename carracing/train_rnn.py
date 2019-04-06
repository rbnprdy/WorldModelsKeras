"""Trains the rnn on data created using `extract.py`."""
import argparse

from models.rnn import get_rnn
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import h5py

def main(args):
    data_path = args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path
    end = args.data_length
    sequence_length = args.sequence_length

    f = h5py.File(data_path, 'r')
    zs = f['z'][:]
    actions = f['action'][:]

    # Combine encoder output and action
    x_train = np.concatenate([zs, actions], axis=-1)
    # Offset x and y
    x_train = x_train[:,:-1]
    y_train = zs[:,1:]

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='train_loss')
    rnn, _ = get_rnn(x_train.shape[1:])
    rnn.compile(loss=rnn.loss, optimizer='adam')
    rnn.fit(x_train, y_train,
	    epochs=epochs,
	    batch_size=batch_size,
	    shuffle='batch',
            callbacks=[checkpoint])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train the rnn.')
    parser.add_argument('--data_path', '-d', default='data/series.h5',
			help='The path to the training data.')
    parser.add_argument('--epochs', '-e', type=int, default=40,
			help='The number of epochs to train for.')
    parser.add_argument('--batch_size', '-b', type=int, default=128,
			help='The batch size to use for training.')
    parser.add_argument('--data_length', '-l', type=int, default=10000,
                        help='The length of input data to use.')
    parser.add_argument('--checkpoint_path', default='checkpoints/rnn.h5',
			help='The path to save the checkpoint at.')
    parser.add_argument('--sequence_length', type=int, default=1000,
                        help='The sequence length to train the rnn on.')
    main(parser.parse_args())
