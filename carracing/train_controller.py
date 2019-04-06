"""Trains the controller on the CarRacing environment."""
import argparse
import random

from env import make_env
from models.vae import get_vae
from models.rnn import get_rnn
from models.controller import get_controller
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import gym


# Taking this code from Karparthy pong tutorial
# (may need to adjust some?)
def discount_rewards(r, gamma):
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r


def main(args):
    epochs = args.epochs
    checkpoint_path = args.checkpoint_path
    max_step = args.max_step
    gamma = args.gamma
    decay = args.decay
    disturbance = args.disturbance
    reward_clip = args.reward_clip

    latent_dim = 32
    vae = get_vae((64, 64, 3), latent_dim)
    vae.load_weights('checkpoints/vae.h5')

    encoder = Model(inputs=vae.input,
                    outputs=vae.get_layer('encoder').output)

    rewrad_dim = 3
    rnn_train, rnn = get_rnn((None, latent_dim + rewrad_dim), train=False)
    rnn_train.load_weights('checkpoints/rnn.h5')

    lstm_dim = 256
    controller = get_controller([latent_dim + lstm_dim], hidden_size=40)
    controller.compile(optimizer=Adam(clipnorm=1.), loss='mse')
    controller.summary()
    env = make_env('carracing', full_episode=True)

    def step(obs, action, h, c, disturbance=0):
        encoded_obs = encoder.predict(np.expand_dims(obs, axis=0))[0]
        obs_and_action = np.concatenate([encoded_obs, action], axis=1)
        h, c = rnn.predict([np.expand_dims(obs_and_action, axis=0), h, c])
        x = np.concatenate([encoded_obs, h], axis=1)
        action = controller.predict(x)
        if random.uniform(0, 1) < disturbance:
            action = np.expand_dims(env.action_space.sample(), axis=0)
        return x, action, h, c

    for epoch in range(epochs):
        obs = env.reset()
        # Take first action
        action = np.expand_dims(env.action_space.sample(), axis=0)
        h = np.zeros((1, lstm_dim))
        c = np.zeros((1, lstm_dim))
        x, action, h, c = step(obs, action, h, c)
        obs, reward, _, _ = env.step(action[0])
        # Keep track of hs, actions, and rewards for training
        x_train = x
        y_train = action
        rewards = [np.clip(reward, -reward_clip, reward_clip)]

        for _ in range(max_step):
            env.render('rgb_array')

            x, action, h, c = step(obs, action, h, c)
            obs, reward, done, _ = env.step(action[0])
            x_train = np.concatenate([x_train, x])
            y_train = np.concatenate([y_train, action])
            clipped_reward = np.clip(reward, -reward_clip, reward_clip)
            rewards.append(clipped_reward)

            if done:
                break

        print('At the end of episode', epoch, 'total reward is', np.sum(rewards))
        controller.fit(x_train,
                       y_train,
                       verbose=1,
                       sample_weight=discount_rewards(rewards, gamma))

        controller.save_weights(checkpoint_path)
        # adjust randomness
        disturbance /= (1 + decay*epoch)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Train the controller.')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='The number of epochs to train for.')
    parser.add_argument('--checkpoint_path',
                        default='checkpoints/controller.h5',
                        help='The path to save the checkpoint at.')
    parser.add_argument('--max_step', type=int, default=2000,
                        help='The maximium number of steps to try before ' +
                        'restarting environemnt.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Gamma value for discounting rewards.')
    parser.add_argument('--disturbance', type=float, default=0.5,
                        help='The initial probability of choosing a random ' +
                        'action')
    parser.add_argument('--decay', type=float, default=0.1,
                        help='The amount to decay the probability of a random ' +
                        'step each epoch.')
    parser.add_argument('--reward_clip', type=float, default=1.,
                        help='How much to clip the rewards by.')
    main(parser.parse_args())
