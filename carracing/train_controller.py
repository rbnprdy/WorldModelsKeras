"""Trains the controller on the CarRacing environment."""
import argparse

from env import make_env
from models.vae import get_vae
from models.rnn import get_rnn
from models.controller import get_controller
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
import numpy as np
import gym


# Taking this code from Karparthy pong tutorial
# (may need to adjust some?)
def discount_rewards(r, gamma):
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = runnig_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r


def main(args):
    epochs = args.epochs
    checkpoint_path = args.checkpoint_path
    max_step = args.max_step
    gamma = args.gamma

    latent_dim = 32
    vae = get_vae((64, 64, 3), latent_dim)
    vae.load_weights('checkpoints/vae.h5')

    encoder = Model(inputs=vae.input,
                    outputs=vae.get_layer('encoder').output)

    rewrad_dim = 3
    rnn_train, rnn = get_rnn((None, latent_dim + rewrad_dim), train=False)
    rnn_train.load_weights('checkpoints/rnn.h5')

    lstm_dim = 256
    controller = get_controller([latent_dim + lstm_dim])

    env = make_env('carracing', full_episode=True)

    def step(obs, h, c):
        encoded_obs = encoder.predict(np.expand_dims(obs, axis=0))[0]
        obs_and_action = np.concatenate([encoded_obs, action], axis=1)
        h, c = rnn.predict([np.expand_dims(obs_and_action, axis=0), h, c])
        action = controller.predict(np.concatenate([encoded_obs, h], axis=1))
        return action, h, c

    for epoch in range(epochs):
        obs = env.reset()
        # Take first action
        h = np.zeros((1, lstm_dim))
        c = np.zeros((1, lstm_dim))
        action, h, c = step(obs, h, c)
        obs, reward, _, _ = env.step(action)
        # Keep track of hs, actions, and rewards for training
        hs = h
        actions = action
        rewards = [reward]
        total_reward = reward

        for _ in range(max_step):
            env.render('rgb_array')

            action, h, c = step(obs, h, c)
            obs, reward, done, _ = env.step(action)
            hs = np.concatenate([hs, h])
            actions = np.concatenate([actions, action])
            rewards.append(reward)
            total_reward += reward

            if done:
                break

        print('At the end of episode', epoch, 'total reward is', total_reward)
        model.fit(hs,
                  actions,
                  verbose=1,
                  callback=[checkpoint],
                  sample_weight=discount_rewards(rewards, gamma))


if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Train the controller.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='The number of epochs to train for.')
    parser.add_argument('--checkpoint_path', default='checkpoints/vae.h5',
                        help='The path to save the checkpoint at.')
    parser.add_argument('--max_steps', type=int, default=3000,
                        help='The maximium number of steps to try before ' +
                        'restarting environemnt.')
    parser.add_argument('--gamma'-, type=float, default=0.99,
                        help='Gamma value for discounting rewards.')
    main(parser.parse_args())
