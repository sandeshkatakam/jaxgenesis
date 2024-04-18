from typing import Any, Optional
import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array
import flax.linen as nn
from typing import Tuple


import json
from ...utils import load_torch_weights
from . import Abstract_GAN

def load_configs(path):
    """Loads the Hyperparaemters from the configuration file"""
    configs = json.load(path)
    return configs

class Generator_Network(nn.Module):
    """Generator class for GAN"""
    def __init__(self, config_path):
        super().__init__()
        self.configs = load_configs(config_path)
        if not key:
            key = jrandom.PRNGKey(0)
        keys = jrandom.split(key,0)

## TODO: Complete base Generator and Discriminator Classes
## This week: Complete Vanilla GAN , InfoGAN, WGAN, DCGAN and write Tests
## Next week: Complete WGAN, CycleGAN etc and write tests
## Third week: VAEs Complete with tests
## Fourth Week: 

class Discriminator_Network(nn.Module):
    """Discriminator class for GAN"""
    features: Tuple[int,...] # Replace this with JAXTyping
    

    def setup(self, features):
        self.features = features
        self.layers = [nn.Conv(features[0],features[1],kernel_size=4,stride=2,padding=1),

    def __call__(self,x):
        return self.forrward(x)


    def forrward():
        """Method defining a forward pass for Discriminator Network"""

class Vanilla_GAN(Abstract_GAN,eqx.Module):
    """A Simple GAN Network"""
    def __init__(self):
        super().__init__()
        self.configs = load_configs(path)



        


    
