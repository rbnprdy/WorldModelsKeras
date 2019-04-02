"""Trains the controller on the CarRacing environment."""
import argparse

from models.vae import get_vae
from models.rnn import get_rnn
from models.controller import get_controller
from tensorflow.keras.callbacks import ModelCheckpoint
import gym

def main(args):

    latent_dim = 32
    vae = get_vae((64, 64, 3), latent_dim)
    vae.load_weights('checkpoints/vae.h5')

    encoder = Model(inputs=vae.input,
                    outputs=vae.get_layer('encoder').output)

    rewrad_dim = 3
    rnn = get_rnn((None, latent_dim + rewrad_dim))

    lstm_dim = 256
    controller = get_controller((latent_dim + lstm_dim))

    env = make_env('carracing', full_episode=True)
    for epoch in range(epochs):
        obs = env.reset()
        while True:
            env.render('rgb_array')
            action = 

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Train the controller.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='The number of epochs to train for.')
    parser.add_argument('--checkpoint_path', default='checkpoints/vae.h5',
                        help='The path to save the checkpoint at.')
    main(parser.parse_args())