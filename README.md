# JaxGenesis: Generative Models Library for JAX Deep Learning Framework



![JAX Logo](./assets/imgs/jaxgenesislogo.png)  

Implementations of different generative model architectures in JAX framework.

JAX is a deep learning framework that enables training of CPU/GPU/TPU. 

## Implementations of Generative model architectures:
* Generative Adversaraial Networks (GAN) Models: 
    * Vanilla-GAN
    * Deep Convolutional GAN(DC-GAN)
    * Conditional GAN (C-GAN)
    * Wasserstein GAN (WGAN)
    * Progressive GAN (ProGAN)
    * InfoGAN
    * AutoEncoders
    * Energy Based GAN(EBGAN)
* Variational Auto-Encoder Models:
    * Variational Auto-Encoder Model
    * Conditional VAE
    * WAE-MMD
    * Categorical VAE
    * Joint VAE
    * Info VAE
    * 
* Flow-Based Models(Normalizing Flows):
    * Planar Flow
    * Neural Spline Flow
    * Residual Flow
    * Stochastic Normalizing Flow
    * Continous Normalizing Flows

* Energy Based Models:
    * Restricted Boltzmann Machine(RBM)
    * Deep Belief Networks(DBN)
* NeuralSDEs (for Continous-Time Generative Models for Time Series Generation)

## Quick Start

### Testing and Inference Mode:
Perform testing using pre-trained GAN Models. The pretrained model weights in `pre_trained/` will be downloaded and generate pictures. 

### Training your own GAN:
You can train your own GAN from scratch with `training/`. To change the parameters of the Model you can tweak the parameters in `config.json` script and run the model.

## Benchamarking on Datasets:
* MNIST
* CIFAR10
* CelebA (64x64)
* CelebA (128x128)

## Results of Pre-Trained Models

## Note:  
This repository will continually updated with new implementation of Generative models. 
This is an ongoing project!!
Refer to CONTRIBUTING.md for more details about contributing to this project



### Citation
```
@misc{sandeshkatakam,
  author = {Sandesh, Katakam},
  title = {JAXGenesis},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sandeshkatakam/jaxgenesis}}
}
```