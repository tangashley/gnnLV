"""
File: utils.py
Author: David Dalton
Description: Utility functions for initialising emulators and saving results
"""

import os

import pathlib
import pickle

from jax import random
import jax.numpy as jnp
import flax

# import models
import attn_gnn_models as models


def create_config_dict(n_shape_coeff, n_epochs, lr, output_dim, enc_dims_before_attn,
                       enc_dims_after_attn, num_heads, theta_enc_dims, layer_norm, rng_seed):
    """Creates dictionary of configuration details for the GNN emulator"""

    return {
        'n_shape_coeff': n_shape_coeff,
        'n_train_epochs': n_epochs,
        'learning_rate': lr,
        'output_dim': output_dim,
        'enc_dims_before_attn': enc_dims_before_attn,
        'enc_dims_after_attn': enc_dims_after_attn,
        'theta_enc_dims': theta_enc_dims,
        'num_heads': num_heads,
        'layer_norm': layer_norm,
        'rng_seed': rng_seed
    }


def save_trained_params(epoch_loss, min_loss, params, epoch_idx, epochs_count, save_dir):
    """Saves the trained parameters of the GNN based on loss value"""

    if epoch_loss < min_loss:
        min_loss = epoch_loss
        with pathlib.Path(save_dir, f'trainedNetworkParams.pkl').open('wb') as fp:
            pickle.dump(params, fp)

    return min_loss


def create_savedir(data_path, n_shape_coeff, n_epochs, lr, dir_label, enc_dims_before_attn,
                   enc_dims_after_attn, num_heads, theta_enc_dims, layer_norm, rng_seed):
    """Create directory where emulation resuls are saved

    The emulator's configuration details are written to the directory name for ease of reference
    """

    save_dir = f'emulationResults/{data_path}/attn_{n_shape_coeff}_{n_epochs}_{lr:.1e}_{enc_dims_before_attn}_{enc_dims_after_attn}_{num_heads}_{theta_enc_dims}_{layer_norm}_{rng_seed}{dir_label}/'
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    return save_dir


def load_trained_params(params_save_dir, params_filename="trainedNetworkParams.pkl"):
    """Load trained parameters of GNN emulator from params_save_dir"""

    params_filename_full = params_save_dir + params_filename
    if not os.path.isfile(params_filename_full):
        raise FileNotFoundError(f'No file at: {params_filename_full}')

    with pathlib.Path(params_filename_full).open('rb') as fp:
        params_load = pickle.load(fp)

    return params_load


def initialise_network_params(data_loader, model, trained_params_dir: str, rng_seed: int):
    """Initialise the parameters of the GNN emulator

    If initialising from scratch, use the ".init" method from Flax

    If initialising from earlier training results, simply read these parameters
    from trained_params_dir
    """

    if trained_params_dir == "None":
        key = random.PRNGKey(rng_seed)
        V_init, theta_init, z_global_init, _ = data_loader.return_index_0()
        params = model.init(key, V_init, theta_init, z_global_init)
        return params
    else:
        trained_params = load_trained_params(trained_params_dir)
        return trained_params


def init_varying_geom_emulator(config_dict: dict, data_loader, fixed_geom: bool, trained_params_dir: str):
    """Initialise GNN emulator (varying geometry data)

    Initialises GNN architecture and trainable paramters for prediction of varying LV geom data

    If trained_params_dir is "None", the parameters are initialised randomly
    If trained_params_dir is a directory path, pre-trained parameters are read from there
    """

    # initialise GNN architecture based on configuration details
    model = models.DeepAttnGraphEmulator(enc_dims_before_attn=config_dict['enc_dims_before_attn'],
                                         enc_dims_after_attn=config_dict['enc_dims_after_attn'],
                                         theta_enc_dims=config_dict['theta_enc_dims'],
                                         num_heads=config_dict['num_heads'],
                                         output_dim=[config_dict['output_dim']],
                                         layer_norm=config_dict['layer_norm'])

    # initialise trainable emulator parameters (either randomly or read from trained_params_dir)
    params = initialise_network_params(data_loader, model, trained_params_dir, config_dict['rng_seed'])

    return model, params


def initialise_emulator(emulator_config_dict, data_loader, trained_params_dir="None"):
    # initialise attn-based GNN model and parameters
    model, params = init_emulator(emulator_config_dict, data_loader, trained_params_dir)

    return model, params


def init_emulator(config_dict, data_loader, trained_params_dir):
    """Initialise attn-based GNN emulator

        Initialises GNN architecture and trainable paramters for prediction of varying LV geom data

        If trained_params_dir is "None", the parameters are initialised randomly
        If trained_params_dir is a directory path, pre-trained parameters are read from there
        """

    # initialise GNN architecture based on configuration details
    model = models.DeepAttnGraphEmulator(enc_dims_before_attn=config_dict['enc_dims_before_attn'],
                                         enc_dims_after_attn=config_dict['enc_dims_after_attn'],
                                         theta_enc_dims=config_dict['theta_enc_dims'],
                                         num_heads=config_dict['num_heads'],
                                         output_dim=[config_dict['output_dim']],
                                         layer_norm=config_dict['layer_norm'])

    # initialise trainable emulator parameters (either randomly or read from trained_params_dir)
    params = initialise_network_params(data_loader, model, trained_params_dir, config_dict['rng_seed'])

    return model, params


def print_error_statistics(Utrue, Upred, logging):
    """Prints prediction error statistics to console
    """

    # calculate point-wise RMSE between true and predicted values
    def rmse(true, pred=0):
        return (((true - pred) ** 2).sum(-1)) ** .5

    # find the average magnitude of the true displacement vectors
    mean_norm = rmse(Utrue).mean()

    # rmse errors between true and predicted displacements
    prediction_errors = rmse(Utrue, Upred)

    # find 25th, 50th (median) and 75th percentile values of the prediction errors
    error_quantiles = jnp.percentile(prediction_errors, jnp.array([25., 50., 75.]))

    # print results to console
    logging.info(f'Mean Displacement Vector Norm: {mean_norm:.2f}')
    logging.info(
        f'Prediction Error Percentiles: 25%:{error_quantiles[0]:.2e}, 50%:{error_quantiles[1]:.2e}, 75%:{error_quantiles[2]:.2e}')
