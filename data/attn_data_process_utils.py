"""
File: data_process_utils.py
Author: David Dalton
Description: Utility functions to process raw simulation data
"""

import os

import numpy as np
from numpy import newaxis

import os
import shutil

import jax
jax.config.update('jax_platform_name', 'cpu')
from jax import vmap
import jax.numpy as jnp

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from functools import partial
from typing import Sequence

from absl import logging

# some strings printed to console to seperate different results sections
SECTION_SEPERATOR = '##############################'
SECTION_SEPERATOR2 = SECTION_SEPERATOR * 2

# we normalise shape coefficients with respect to the variance on the below indexed column ## Why?
# REFERENCE_SHAPE_COEFF_COLUMN = 1

####################################
## Augmented Graph Topology Generation
####################################

def attn_node_feature(data_dir, data_types):
    for data_type_i in data_types:
        logging.info(SECTION_SEPERATOR)
        logging.info(f'{data_type_i} data:')
        node_feature = jnp.load(f'{data_dir}/rawData/{data_type_i}/real-node-features.npy')
        node_coords = jnp.load(f'{data_dir}/rawData/{data_type_i}/real-node-coords.npy')
        attn_node_feature = jnp.concatenate((node_feature, node_coords), axis=-1)
        np.save(f'{data_dir}/rawData/{data_type_i}/attn-node-features.npy', attn_node_feature)




####################################
## Data Normalisation
####################################

def compute_norm_stats(data_array: jnp.array, file_type: str):
    """Computes mean and value of inputted data

    Parameters:
    -----------
    data_array: jnp.array
        Array of data to be normalised
    file_type: str
        Name of file - for 'shape-coeffs', computing normalisation
        statistics is handled differently

    Returns:
    ----------
    mean_val, std_val
        Mean and standard deviation of data_array
    """


    # select axes to normalise over
    if len(data_array.shape) == 3:
        axis = (0,1)
    elif len(data_array.shape) == 2:
        axis = 0
    else:
        axis = 0

    mean_val = data_array.mean(axis=axis)

    # if file_type != 'shape-coeffs':
    #     std_val = data_array.std(axis=axis)
    # else:
    #     col_num = REFERENCE_SHAPE_COEFF_COLUMN
    #     std_val = data_array[:,col_num].std().reshape(-1)
    # Different from the Original Codes
    std_val = data_array.std(axis=axis)

    logging.info(f'{file_type} mean: {mean_val}')
    logging.info(f'{file_type} std: {std_val}\n')

    return mean_val, std_val

def normalise_data_array(data_array: jnp.array, data_type: str, file_type: str, summary_stats_dir: str, existing_summary_stats_dir: str = "None"):
    """Normalises inputted data to mean zero unit variance (column-wise)

    Parameters:
    -----------
    data_array: jnp.array
        Array of data to be normalised
    data_type: str
        One of "train", "validation" or "test"
    file_type: str
        Name of file - for 'shape-coeffs', computing normalisation
        statistics is handled differently
    summary_stats_dir: str
        Directory where normalisation statistics will be saved
    exising_summary_stats_dir: str
        Directory where already computed statistics are saved
        (used when normalising fixed geometry data for transfer learning)

    Returns:
    ----------
    data_array_norm: jnp.array
        data_array where columns are normalsied
    """

    if (data_type == 'train') and (existing_summary_stats_dir == "None"):
        mean_val, std_val = compute_norm_stats(data_array, file_type)
        jnp.save(f'{summary_stats_dir}/{file_type}-mean.npy', mean_val)
        jnp.save(f'{summary_stats_dir}/{file_type}-std.npy', std_val)
    else:
        mean_val = jnp.load(f'{summary_stats_dir}/{file_type}-mean.npy')
        std_val = jnp.load(f'{summary_stats_dir}/{file_type}-std.npy')

    # normalise data
    data_array_norm = (data_array - mean_val) / std_val

    # sanity check to mak sure normlisation worked
    _ =  compute_norm_stats(data_array_norm, f'{file_type}-normalised')

    return data_array_norm

def generate_normalised_data(data_dir: str, existing_summary_stats_dir: str, data_types, copy_filenames, normalise_filenames):
    """Normalises processes simulation data before being used for emulation

    Parameters:
    -----------
    data_dir: str
        Name of directory in /data where simulation data is saved
    exising_summary_stats_dir: str
        Directory where already computed statistics are saved
        (used when normalising fixed geometry data for transfer learning)
    data_types: Sequence[str]
        Generally =  ["train", "validation", "test"]
    copy_filenames: Sequence[str]
        List of files which just need to be copied to the processedData subdirectory
        (like node coordinates)
    normalise_filenames: Sequence[str]
        List of files that need to be normalised before being saved to the processedData
        subdirectory (like node features)

    Returns:
    ----------
    Nothing is returned, but all files are processed and saved to the
    subdirectory /processedData inside /data_dir, which can then be used
    to perform emulation
    """

    logging.info(SECTION_SEPERATOR2)
    logging.info('Calling generate_normalised_data()')
    logging.info(SECTION_SEPERATOR2 + '\n')

    # directory to hold the normalisation statistics for the data
    summary_stats_dir = f'{data_dir}/normalisationStatistics'

    # if passed in an existing summary stats dir, copy this to new directory for current data
    if existing_summary_stats_dir == "None":
        if not os.path.isdir(summary_stats_dir): os.mkdir(summary_stats_dir)
    else:

        if not os.path.isdir(existing_summary_stats_dir):
            raise NotADirectoryError(f'No directory at: {existing_summary_stats_dir}')

        logging.info(f'Copying {existing_summary_stats_dir} to {summary_stats_dir}')
        shutil.copytree(existing_summary_stats_dir, summary_stats_dir)

    normalise_data_partial = partial(normalise_data_array,
                                     summary_stats_dir = summary_stats_dir,
                                     existing_summary_stats_dir = existing_summary_stats_dir)

    for data_type_i in data_types:

        logging.info(SECTION_SEPERATOR)
        logging.info(f'Saving normalised {data_type_i} data\n')

        # initialise save directory
        data_type_i_savedir = f'{data_dir}/processedData/{data_type_i}'
        if not os.path.isdir(data_type_i_savedir): os.makedirs(data_type_i_savedir)

        # copy across each file in "copy_filenames"
        for copy_file_j in copy_filenames:
            shutil.copy(f'{data_dir}/rawData/{data_type_i}/{copy_file_j}.npy', data_type_i_savedir)

        # normalise and save each file in "normalise_filenames"
        for file_type_j in normalise_filenames:

            # load un-normalised data
            data_array = jnp.load(f'{data_dir}/rawData/{data_type_i}/{file_type_j}.npy')

            if file_type_j=='real-node-displacement' and (data_type_i in ['validation', 'test']):
                # we do not normalise validation / test displacement values
                data_array_norm = data_array
            else:
                # normalise data
                data_array_norm = normalise_data_partial(data_array, data_type_i, file_type_j)

            # save normalised data
            jnp.save(f'{data_type_i_savedir}/{file_type_j}.npy', data_array_norm)


