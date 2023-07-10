"""
File: models.py
Author: David Dalton
Description: Implements DeepGraphEmulator GNN Architecture
"""
import flax.linen
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np

from flax import linen as nn

from typing import Sequence, Callable

from gnnLV import models as gnnLV_models

DTYPE = jnp.float32
from jax import random

class FlaxAttnEncoder(nn.Module):
    """Implements a self-attention encoder in Flax"""
    enc_dims_before_attn: Sequence[int]
    enc_dims_after_attn: Sequence[int]
    layer_norm: bool
    num_heads: int

    # def setup(self):
    #     self.MLP_before_attn = [nn.Dense(dim, dtype=DTYPE) for dim in self.enc_dims_before_attn]
    #     self.attn = flax.linen.SelfAttention(self.num_heads)
    #     self.MLP_after_attn = [nn.Dense(dim, dtype=DTYPE) for dim in self.enc_dims_after_attn]
    @nn.compact
    def __call__(self, inputs):
        x = inputs

        # process data before they enter the attn layer
        for dim in self.enc_dims_before_attn[:-1]:
            x = nn.Dense(dim, dtype=DTYPE)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)

        # attn layer
        x = flax.linen.SelfAttention(self.num_heads[0])(x)

        # process data obtained from attn layer
        for dim in self.enc_dims_before_attn[:-1]:
            x = nn.Dense(dim, dtype=DTYPE)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)

        return x


# class FlaxAttnDecoder(nn.Module):
#     def setup(self, layer_dims, layer_norm):
#

class DeepAttnGraphEmulator(nn.Module):
    """DeepGraphEmulator (varying geometry data)"""
    enc_dims_before_attn: Sequence[int]  # dimensions for MLPs before self-attn layer
    enc_dims_after_attn: Sequence[int]  # dimensions for MLPs after self-attn layer
    theta_enc_dims: Sequence[int]  # dimensions for MLP for global information
    layer_norm: bool
    num_heads: int
    output_dim: int

    # def setup(self, enc_dims_before_attn, enc_dims_after_attn, theta_enc_dims, num_heads, output_dim, layer_norm):
    #
    #     self.enc_dims_before_attn = enc_dims_before_attn  # dimensions for MLPs before self-attn layer
    #     self.enc_dims_after_attn = enc_dims_after_attn  # dimensions for MLPs after self-attn layer
    #     self.theta_enc_dims = theta_enc_dims  # dimensions for MLP for global information
    #     self.layer_norm = layer_norm
    #     self.num_heads = num_heads
    #     self.output_dim = output_dim

    @nn.compact
    def __call__(self, V: jnp.ndarray,
                 # E: jnp.ndarray,
                 theta: jnp.ndarray,
                 z_global: jnp.ndarray = None,
                 sow_latents: bool = False):

        """

        Inputs:
        ---------
        V: jnp.ndarray
            Array giving feature vectors of each node (real and virtual)
        # E: jnp.ndarray
        #     Array giving feature vectors of each edge
        theta: jnp.ndarray
            Vector of global graph parameters
        z_global: jnp.ndarray (Optional)
            Vector embedding of the global shape of the geometry
        sow_latents: bool
            Boolean controlling the output returned: see below for details

        Outputs:
        ---------
        U: jnp.ndarray (if sow_latents = False)
            Array of displacement predictions for each real node in V
        z_local: jnp.ndarray (if sow_latents = True)
            Array of latent embeddings $z_i^{local}$ for each real node $i$ in V,
            found by the Processor stage of the emulator.
        """

        ## Initialise encoder with self-attention:
        node_attn_enc = FlaxAttnEncoder(self.enc_dims_before_attn, self.enc_dims_after_attn,
                                        self.layer_norm, self.num_heads)

        # Initialise 1 + D decoder MLPs
        theta_encode_mlp = gnnLV_models.make_layernorm_mlp(self.theta_enc_dims[0])
        node_decode_mlps = [gnnLV_models.make_mlp(self.enc_dims_after_attn + (1,)) for _ in range(self.output_dim[0])]

        ## Encoder:
        V = node_attn_enc(V)

        # final local learned representation is a concatenation of vector embedding and incoming messages
        z_local = V

        # save value of final learned representation if required for fixed geometry emulator
        if sow_latents:
            return z_local

        ## Decoder:
        # encode global parameters theta
        z_theta = theta_encode_mlp(theta)

        # tile global values (z_theta and optionally z_global) to each individual real node
        if z_global is None:
            globals_array = jnp.tile(z_theta, (z_local.shape[0], 1))
        else:
            # stack z_global with z_theta if z_global is inputted
            print("z_theta.shape" + str(z_theta.shape))
            print("z_local.shape" + str(z_local.shape))
            global_embedding = jnp.hstack((z_theta, z_global.reshape(1, -1)))
            globals_array = jnp.tile(global_embedding, (z_local.shape[0], 1))
            print("globals_array.shape" + str(globals_array.shape))


        # final learned representation is (z_theta, z_local) or (z_theta, z_global, z_local)
        final_representation = jnp.hstack((globals_array, z_local))

        # make prediction for forward displacement using different decoder mlp for each dimension
        individual_mlp_predictions = [decode_mlp(final_representation) for decode_mlp in node_decode_mlps]

        # concatenate the predictions of each individual decoder mlp
        Upred = jnp.hstack(individual_mlp_predictions)

        # return displacment prediction array
        return Upred




if __name__ == "__main__":
    x = jnp.array(np.asarray([
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]] # 1 sample, with 3 points
    ]), dtype=jnp.float32)
    z_theta = jnp.array(np.asarray([[10, 10, 10]]), dtype=jnp.float32)
    z_global = jnp.array(np.asarray([[0.1, 0.1, 0.1]]), dtype=jnp.float32)
    model = DeepAttnGraphEmulator(enc_dims_before_attn=(128, 128),
                                  enc_dims_after_attn=(40),
                                  theta_enc_dims=(128, 128, 40),
                                  num_heads=1,
                                  output_dim=[2],
                                  layer_norm=True)
    key = random.PRNGKey(0)
    params = model.init(key, x[0], z_theta[0:1], z_global[0])
    pred = model.apply(params, x[0], z_theta[0:1], z_global[0])
    print()
