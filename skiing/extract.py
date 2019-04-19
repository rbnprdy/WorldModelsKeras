"""Generates data to train the vae model. Code is adapted from
https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/extract.py

Usage:
    python extract.py OUTPUT_DIR --num_frames NUM_FRAMES --num_trials NUM_TRIALS
"""

import os
import argparse
import random

import gym
import numpy as np

from env import SkiingWrapper


def main(args):
    output_dir = args.output_dir
    num_frames = args.num_frames
    num_trials = args.num_trials

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if output_dir.endswith('/'): output_dir = output_dir[:-1]

    env = gym.make('Skiing-v0')
    env = SkiingWrapper(env)

    # Generate `num_trials` trials
    for trial in range(num_trials):
        random_generated_int = random.randint(0, 2**31-1)
        filename = output_dir + "/" + str(random_generated_int) + ".npz"
        recording_obs = []
        recording_action = []

        np.random.seed(random_generated_int)
        env.seed(random_generated_int)
        
        # Intial observation and action
        obs = env.reset()

        # Generate `num_frames` frames per trial
        for frame in range(num_frames):
            recording_obs.append(obs)
            
            action = env.action_space.sample()
            recording_action.append(action)

            # We shouldn't finish in a thousand frames, but if we do,
            # we need to start over so that all testing arrays are same size
            obs, reward, done, _ = env.step(action)
            if done:
                print('[WARNING] Environment finished early in seed', random_generated_int)
                obs = env.reset()

        recording_obs = np.array(recording_obs, dtype=np.uint8)
        recording_action = np.array(recording_action, dtype=np.uint8)
        np.savez_compressed(filename, obs=recording_obs, action=recording_action)

    env.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Generate training data for the vae model.')
    parser.add_argument('output_dir',
                        help='The directory to place the output data in.')
    parser.add_argument('--num_frames', type=int, default=1000,
                        help='The number of frames for one episode.')
    parser.add_argument('--num_trials', type=int, default=200,
                        help='The number of trials to run.')
    main(parser.parse_args())
