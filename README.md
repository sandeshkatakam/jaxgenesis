# Generative Models in JAX Deep Learning Framework
Implementations of different generative model architectures in JAX framework using Haiku(DeepMind) Library for Neural Network Implementations.

## JAX - Accelerated Deep Learning Framework

![JAX Logo](assets/imgs/logo_jax.jpeg)
JAX is a deep learning framework that enables training of CPU/GPU/TPU. 

## Implementations of Generative model architectures:
* Generative Adversaraial Networks (GANs) 
    * Vanilla-GAN
    * Deep Convolutional GAN(DC-GAN)
    * Conditional GAN (C-GAN)
    * Wasserstein GAN (WGAN)
    * Progressive GAN (ProGAN)
* Variational Auto-Encoders
    * Auto-Encoder Model architecture
    * 
* Flow-Based Models
* Energy Based Models
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
