__all__ = ["BaseTrainer"]
import abc
import collections
import typing
import torch  
from torch.datasets import data_loader

from abc import ABC, abstractmethod






class AbstractTrainer(ABC):
    ## Abstract Trainer Class for GAN
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def load_checkpoint(self):
        pass

    @abstractmethod
    def save_history(self):
        pass

    @abstractmethod
    def load_history(self):
        pass




class BaseTrainer(AbstractTrainer):
    """
    Base GAN Trainer Class 
    All other different variations of GAN Trainers should subclass this class
    
    Args:
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        optimizer (Optimizer): The optimizer used for training.
        callbacks (list): List of callbacks to be executed during training.
        save_model (bool): Flag indicating whether to save the trained model.
    """
    def __init__(self, epochs, learning_rate, optimizer, callbacks=None, save_model=False):
        super().__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.save_model = save_model
        
    def train():
        pass