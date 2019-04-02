"""Generates data to train the vae model. Code is adapted from
https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/extract.py

Usage:
    python extract.py OUTPUT_DIR --max_frames MAX_FRAMES --max_trials MAX_TRIALS
"""

import os
import argparse
import random
import time

import gym
import numpy as np
import h5py

from env import make_env

def main(args):
    output_file = args.output_file
    max_frames = args.max_frames
    max_trials = args.max_trials

    if not os.path.exists(os.path.split(output_file)[0]):
        os.makedirs(os.path.split(output_file)[0])

    f = h5py.File(output_file, 'w')
    dset_obs = f.create_dataset('obs', shape=(max_trials*max_frames, 64, 64, 3))
    dset_action = f.create_dataset('action', shape=(max_trials*max_frames, 3))

    env = make_env('carracing', full_episode=True)
    total_frames = 0
    for trial in range(max_trials):
        try:
            start = time.time()
            random_generated_int = random.randint(0, 2**31-1)
            recording_obs = []
            recording_action = []

            np.random.seed(random_generated_int)
            env.seed(random_generated_int)

            obs = env.reset()

            for frame in range(max_frames):
                env.render("rgb_array")
                recording_obs.append(obs)
                action = env.action_space.sample()
                recording_action.append(action)

                obs, reward, done, info = env.step(action)

                if done:
                    break

            total_frames += (frame+1)
            print("dead at", frame+1, "total recorded frames for this worker", total_frames)
            recording_obs = np.array(recording_obs, dtype=np.uint8)
            recording_action = np.array(recording_action, dtype=np.float16)
            dset_obs[trial*max_frames:(1+trial)*max_frames] = recording_obs
            dset_action[trial*max_frames:(1+trial)*max_frames] = recording_action
            print(time.time() - start)
        except gym.error.Error:
            print("stupid gym error, life goes on")
            env.close()
            env = make_env(render_mode=render_mode)
            continue

    env.close()
    f.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Generate training data for the vae model.')
    parser.add_argument('--output_file', '-o', default='data/train.h5',
                        help='The directory to place the output data in.')
    parser.add_argument('--max_frames', type=int, default=1000,
                        help='The max frames for one episode.')
    parser.add_argument('--max_trials', type=int, default=200,
                        help='The max number of trials to run.')
    main(parser.parse_args())
