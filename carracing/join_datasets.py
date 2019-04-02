"""Joins a set of hdf5 datasets into one dataset."""
import argparse
import os
import random

import h5py
import numpy as np

DATA_SHAPE = (64, 64, 3)
ACTION_SHAPE = 3
TOTAL_NUM = 12800000
NUM_PER_FILE = 1000*200

def main():
    parser = argparse.ArgumentParser(
        description='Joins a set of hdf5 datasets into one dataset.')
    parser.add_argument('--data_path', '-d', default='data/',
                        help='Path to the folder with the data to join.')
    parser.add_argument('--output_file', '-o', default='data/train.h5',
                        help='The file to save the output data to.')
    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.output_file

    file_list = os.listdir(data_path)
    if not os.path.exists(os.path.split(output_path)[0]):
        os.makedirs(os.path.split(output_path)[0])

    with h5py.File(output_path, 'w') as output_file:
        obs = output_file.create_dataset('obs',
                                         shape=(TOTAL_NUM, *DATA_SHAPE),
                                         dtype=np.uint8)
        action = output_file.create_dataset('action',
                                            shape=(TOTAL_NUM, ACTION_SHAPE),
                                            dtype=np.float16)

        for i, f in enumerate(file_list):
            print('joining file', i, 'of', len(file_list), end='\r')
            with h5py.File(os.path.join(data_path, f), 'r') as input_file:
                obs_in = input_file.get('obs')
                action_in = input_file.get('action')
                obs[i*NUM_PER_FILE:(i+1)*NUM_PER_FILE] = obs_in
                action[i*NUM_PER_FILE:(i+1)*NUM_PER_FILE] = action_in


if __name__=='__main__':
    main()
