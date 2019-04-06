import argparse
import random
import time

import h5py
import numpy as np
from sklearn.utils import shuffle


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    batch_size = args.batch_size

    with h5py.File(output_path, 'w') as output_file:
        with h5py.File(input_path, 'r') as input_file:
            obs_in = input_file.get('obs')
            action_in = input_file.get('action')

            obs_out = output_file.create_dataset('obs',
                                                 shape=obs_in.shape,
                                                 dtype=np.uint8)
            action_out = output_file.create_dataset('action',
                                                    shape=action_in.shape,
                                                    dtype=np.float16)

            for batch in range(0, len(obs_in), batch_size):
                start = time.time()
                obs_out[batch:batch+batch_size], \
                        action_out[batch:batch+batch_size] = \
                        shuffle(obs_in[batch:batch+batch_size],
                                action_in[batch:batch+batch_size])
                print(time.time() - start, batch, '/', len(obs_in), end='\r')

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Randomizes a dataset and saves it to a new file.')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('--batch_size', type=int, default=100000)
    main(parser.parse_args())
