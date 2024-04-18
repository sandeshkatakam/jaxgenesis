from . import (
    vanillagan,
    dcgan,
    cyclicgan,
)
from conditional_gan import *
from dcgan import *
from  infogan import *
from wgan import *
from vanillagan import *
from flax.linen import nn
import jaxtyping


import abc
from abc import abstractclassmethod, abstractproperty

class Abstract_GAN(abc):
    """Abstract GAN Class """
    def __init__(self):    
        super().__init__()
        pass

    def __call__(self):
        pass
    
    def __iter__(self):
        pass


class AbstractGenerator(nn.Module):
    def __init__(self,config_path):
        super().__init__()
        self.configs = load_configs(config_path)

    def __call__(self):
        pass            

    def __iter__(self):
        pass

class AbstractDiscriminator(nn.Module):
    def __init__(self,config_path):
        super().__init__()
        self.configs = load_configs(config_path)

    def __call__(self):
        pass            

    def __iter__(self):
        pass


        


## NOTE: Perform isort on the package imports to re-order them