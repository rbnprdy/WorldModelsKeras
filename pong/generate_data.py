"""Generates data to train the vae and rnn models

Usage:
python generate_data.py OUTPUT_DIR --num_images NUM_IMAGES"""

import os
import random
import argparse

import gym
import numpy as np

IMAGE_SHAPE = [210, 160, 3]

UP_ACTION = 2
DOWN_ACTION = 3

def main():
    parser = argparse.ArgumentParser(description='Generate data to train vae and rnn models.')
    parser.add_argument('output_dir', help='The directory to place the output data in.')
    parser.add_argument('--num_images', type=int, default=10000, help='The number of images to generate.')
    args = parser.parse_args()
    output_dir = args.output_dir
    num_images = args.num_images

    env = gym.make("Pong-v0")
    env.reset()

    data_shape = tuple([num_images] + IMAGE_SHAPE)
    data = np.zeros(data_shape)

    for i in range(num_images):
        print('Image {}/{}'.format(i, num_images), end='\r')
        action = random.randint(UP_ACTION, DOWN_ACTION)
        observation, _, episode_done, _ = env.step(action)
        data[i] = observation

        if episode_done:
            env.reset()

    print('Saving images...')
    np.save(os.path.join(output_dir, 'pong.npy'), data)

if __name__=='__main__':
    main()
