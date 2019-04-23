"""Utility functions to help training."""
import os
from random import shuffle

import numpy as np

def generate_observation_data(data_dir,
                              batch_size,
                              num_episodes,
                              num_frames,
                              offset=0):
    """Generates data from a directory full of numpy zip files. Numpy files
    must contain an ['obs'] array.
    
    Arguments:
        data_dir -- The path to the directory containing the data.
        batch_size -- The batch size to use.
        num_episodes -- The total number of episodes to generate from.
        num_frames -- The number of frames to use for each episode.
    
    Keyword Arguments:
        offset -- The amount to offset the first file read by. Note taht
                  files are sorted by name before being cut to
                  `num_episodes` length. (default: {0})
    """
    assert num_frames % batch_size == 0, \
           'num_frames must be divisible by batch_size because I am lazy'
    
    filelist = os.listdir(data_dir)
    filelist.sort()
    filelist = filelist[offset:offset+num_episodes]
    shuffle(filelist)
    
    file_num = 0
    image_num = 0    
    
    curr_file = np.load(
        os.path.join(data_dir, filelist[file_num])
    )['obs'].astype(np.float) / 255.
    while True:
        batch = curr_file[image_num:image_num+batch_size]
        np.random.shuffle(batch)

        image_num += batch_size
        if image_num == num_frames:
            image_num = 0
            file_num += 1
            if file_num == len(filelist):
                file_num = 0
                shuffle(filelist)
            curr_file = np.load(
                os.path.join(data_dir, filelist[file_num])
            )['obs'].astype(np.float) / 255.

        yield (batch, batch)

