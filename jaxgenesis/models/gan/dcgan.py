import jax.numpy as jnp
import jax.nn as nn
import equinox as eqx

__all__ = ["Generator", "Discriminator"]

class Generator(eqx.Module):
    r"""Base class for all Generator models
    All Generator models for all the other GANs must subclass this

    Args:
        encoding_dims (int): Dimensions of the sample from the noise prior
        label_type (str, optional): The type of labels expected by the Generator. The available choices are:
            'None' if no label is needed, 'required' if the original labels are
            needed and 'generated' if labels are to be sampled from a distribution"""
            def __init__(self, encoding_dims, label_type = "none"):
                super(Generator, self).__init__()
                self.encoding_dims = encoding_dims
                self.label_type = label_type

            def _weight_initializer(self):
                r"""Default weight initializer for all generator models.
                Models that require custom weight initializer can override this method"""
                for m in self.modules():
                    if isinstance(m, nn.ConvTranspose2d):
                        nn.init.kaiming_normal_(m.weight)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1.0)
                        nn.init.constant_(n.bias, 0.0)
            
            def sampler(self, sample_size, device):
                r"""Fucntion to allow sampling data at inference time. Models requiring input in any other format must override it in the subclass.

                Args:
                    sample_size (int): The number of images to be generated
                    device (torch.device): The device on which data must be generated
                    Returns:
                        A list of the items required as input"""
                        return [torch.randn(sample_size, self.encoding_dims, device = device)]

