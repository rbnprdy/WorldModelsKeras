"""Trains the rnn on data created using `extract.py`."""
import argparse
from models.rnn import get_rnn
from tensorflow.keras.utils import HDF5Matrix


def main():
    parser = argparse.ArgumentParser(description='Train the rnn.')
    parser.add_argument('--data_path', '-d', default='data/train_rnn.h5',
			help='The path to the training data.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
			help='The number of epochs to train for.')
    parser.add_argument('--batch_size', '-b', type=int, default=128,
			help='The batch size to use for training.')
    parser.add_argument('--checkpoint_path', default='checkpoints/rnn.h5',
			help='The path to save the checkpoint at.')
    args = parser.parse_args()
    data_path = args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path
    data_shape = (64, 64, 3)
    rnn = get_rnn(data_shape, 32)
    zs = HDF5Matrix(data_path, 'z')
    actions = HDF5Matrix(data_path, 'action')
    x_train = np.vstack(zs, actions)[:-1]
    y_train = zs[:-1]
    rnn.compile(optimizer='adam')
    rnn.fit(x_train, y_train
	    epochs=epochs,
	    batch_size=batch_size,
	    shuffle='batch')
    rnn.save_weights(checkpoint_path)


if __name__=='__main__':
    main()
