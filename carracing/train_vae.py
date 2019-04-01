"""Trains the vae on data created using `extract.py`."""
import argparse

from models.vae import get_vae
from tensorflow.keras.utils import HDF5Matrix
from tensorflow.keras.callbacks import ModelCheckpoint


def scale(a):
    return a / 255


def main():
    parser = argparse.ArgumentParser(description='Train the vae.')
    parser.add_argument('--data_path', '-d', default='data/train.h5',
                        help='The path to the training data.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='The number of epochs to train for.')
    parser.add_argument('--batch_size', '-b', type=int, default=128,
                        help='The batch size to use for training.')
    parser.add_argument('--checkpoint_path', default='checkpoints/vae.h5',
                        help='The path to save the checkpoint at.')
    parser.add_argument('--log_dir', default='logs/vae/',
                        help='The log directory for tensorboard.')
    parser.add_argument('--train_size', type=int, default=1000000,
                        help='The number of images to use for training.')
    args = parser.parse_args()
    data_path = args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path
    log_dir = args.log_dir
    train_size = args.train_size

    data_shape = (64, 64, 3)
    vae = get_vae(data_shape, 32)

    x_train = HDF5Matrix(data_path, 'obs', end=train_size, normalizer=scale)

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='train_loss')

    vae.compile(optimizer='adam')
    vae.fit(x_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle='batch',
            callbacks=[checkpoint])


if __name__=='__main__':
    main()
