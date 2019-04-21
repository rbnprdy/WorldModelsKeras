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

from env import make_env

def generate_data_action(t, current_action):
    """A bit of a hack to generate better input data taken from
    https://github.com/AppliedDataSciencePartners/WorldModels/blob/master/config.py
    This will go straight for the first 60 time steps, then only either
    accelerate, decelearte, or turn"""
    if t < 60:
        return np.array([0,1,0])
    
    if t % 5 > 0:
        return current_action

    rn = random.randint(0,9)
    if rn in [0]:
        return np.array([0,0,0])
    if rn in [1,2,3,4]:
        return np.array([0,random.random(),0])
    if rn in [5,6,7]:
        return np.array([-random.random(),0,0])
    if rn in [8]:
        return np.array([random.random(),0,0])
    if rn in [9]:
        return np.array([0,0,random.random()])

def main(args):
    output_dir = args.output_dir
    num_frames = args.num_frames
    num_trials = args.num_trials

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if output_dir.endswith('/'): output_dir = output_dir[:-1]

    env = make_env('carracing', full_episode=True)

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
        action = env.action_space.sample()

        # Generate `num_frames` frames per trial
        for frame in range(num_frames):
            env.render("rgb_array")
            recording_obs.append(obs)

            # Is this the problem?
            # action = generate_data_action(frame, action)
            action = env.action_space.sample()
            recording_action.append(action)

            # We're not using done because the agent should never finish
            # so that each trial has the same amount of frames. This is a
            # pretty reasonable assumption because we're acting randomly.
            obs, reward, _, _ = env.step(action)

        recording_obs = np.array(recording_obs, dtype=np.uint8)
        recording_action = np.array(recording_action, dtype=np.float16)
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
