from . import (
    vanillagan,
    dcgan,
    cyclicgan,
)
import abc
from abc import abstractclassmethod, abstractproperty

class Abstract_GAN(abc):
    """Abstract GAN Class """
    def __init__(self):
        pass

    def __call__(self):
        pass
    
    def __iter__(self):
        pass
