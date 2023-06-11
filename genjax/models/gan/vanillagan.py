from typing import Any, Optional
import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array
import json
from ...utils import load_torch_weights
from . import Abstract_GAN

def load_configs(path):
    """Loads the Hyperparaemters from the configuration file"""
    configs = json.load(path)
    return configs

class Generator_Network(eqx.Module):
    """Generator class for GAN"""
    def __init__(self, config_path):
        super().__init__()
        self.configs = load_configs(config_path)
        if not key:
            key = jrandom.PRNGKey(0)
        keys = jrandom.split(key,0)

class Discriminator_Network(eqx.Module):
    """Discriminator class for GAN"""
    def __init__(self, config_path):
        super().__init__()
        self.configs = load_configs(config_path)

    def forrward():
        """Method defining a forward pass for Discriminator Network"""

class Vanilla_GAN(Abstract_GAN,eqx.Module):
    """A Simple GAN Network"""
    def __init__(self):
        super().__init__()
        self.configs = load_configs(path)

        

