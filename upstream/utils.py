#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio DistilBERT Utilities Module
================================

Core utility functions for model manipulation and visualization in the Audio DistilBERT
framework. This module provides essential tools for model pruning, activation functions,
and spectrogram visualization capabilities.

Key Features:
- Attention head pruning for model compression
- GELU activation implementation
- Spectrogram visualization tools
- Model weight manipulation utilities

Author: fanfan-yu
Date: 2025.10.05

Functions:
    prune_linear_layer: Remove specific neurons/channels from linear layers
    gelu: Gaussian Error Linear Unit activation function
    plot_spectrogram_to_numpy: Convert spectrogram to visualizable format
    parse_prune_heads: Parse head pruning configuration
"""

import math

import torch
from torch import nn
import matplotlib.pylab as plt
import numpy as np


def prune_linear_layer(layer, index, dim=0):
    """
    Prune specific neurons or channels from a linear layer.
    
    This function removes specified neurons or channels from a linear layer
    while maintaining the layer's functionality. It's primarily used for
    attention head pruning in transformer models.
    
    Args:
        layer (nn.Linear): The linear layer to prune
        index (torch.Tensor): Indices of neurons/channels to keep
        dim (int): Dimension to prune (0 for weight matrix rows, 1 for columns)
        
    Returns:
        nn.Linear: New pruned linear layer with requires_grad=True
        
    Example:
        >>> # Remove first 3 attention heads from a linear layer
        >>> layer = nn.Linear(768, 768)
        >>> index = torch.arange(768)[3*64:]  # Keep heads after 3rd
        >>> pruned_layer = prune_linear_layer(layer, index)
    """
    index = index.to(layer.weight.device)
    
    # Select weights to keep
    W = layer.weight.index_select(dim, index).clone().detach()
    
    # Handle bias selection
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()  # No change for output layer
        else:
            b = layer.bias[index].clone().detach()
    
    # Create new layer structure
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    
    new_layer = nn.Linear(
        new_size[1], new_size[0], 
        bias=layer.bias is not None
    ).to(layer.weight.device)
    
    # Copy weights with proper handling
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
        
    return new_layer


def gelu(x):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    
    GELU provides a smooth, differentiable approximation of ReLU
    by modeling it as a probabilistic combination of identity and zero.
    This activation is used throughout transformer architectures.
    
    Mathematical Formula:
        GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: GELU-activated tensor
        
    Notes:
        This differs slightly from OpenAI GPT's GELU implementation:
        0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


###############################
# SPECTROGRAM VISUALIZATION #
###############################

def plot_spectrogram_to_numpy(spectrogram):
    """
    Convert spectrogram to numpy array for visualization.
    
    This function creates a visual representation of spectrogram data
    suitable for tensorboard logging or debugging purposes.
    
    Args:
        spectrogram (torch.Tensor or numpy.ndarray): 
            Input spectrogram with shape (freq_bins, time_steps)
            
    Returns:
        numpy.ndarray: RGB image array with shape (3, height, width)
        
    Example:
        >>> spec = torch.randn(80, 100)  # 80-channel, 100-frame spectrogram
        >>> viz = plot_spectrogram_to_numpy(spec)
        >>> print(viz.shape)  # (3, 300, 1200) - RGB image
    """
    spectrogram = spectrogram.transpose(1, 0)
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Create heatmap visualization
    im = ax.imshow(spectrogram, 
                   aspect="auto", 
                   origin="lower",
                   interpolation='none')
    
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    # Convert plot to numpy array
    fig.canvas.draw()
    data = _save_figure_to_numpy(fig)
    plt.close()
    return data


def _save_figure_to_numpy(fig):
    """
    Convert matplotlib figure to numpy array.
    
    Internal utility function that converts a matplotlib figure object
    into a numpy array suitable for tensorboard logging.
    
    Args:
        fig (matplotlib.figure.Figure): The figure to convert
        
    Returns:
        numpy.ndarray: RGB array with shape (3, height, width)
    """
    # Save figure to numpy array
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data.transpose(2, 0, 1)  # Convert to (Channel, Height, Width)


#############################
# ATTENTION HEAD PRUNING #
#############################

def parse_prune_heads(config):
    """
    Parse head pruning configuration from transformer config.
    
    This function processes the head pruning specification from the
    configuration file and converts it into a list of head indices
    to be pruned from the model.
    
    Supported formats:
    - Single indices: "3"
    - Ranges: "12-15"
    - Combinations: "0,1,2,12-15"
    
    Args:
        config (dict): Transformer configuration dictionary
        
    Example:
        >>> config = {'transformer': {'prune_headids': '0,1,2,12-15'}}
        >>> parse_prune_heads(config)
        >>> print(config['transformer']['prune_headids'])
        [0, 1, 2, 12, 13, 14, 15]
    """
    if 'prune_headids' in config['transformer'] and \
       config['transformer']['prune_headids'] != 'None':
        
        heads_int = []
        spans = config['transformer']['prune_headids'].split(',')
        
        # Parse each span specification
        for span in spans:
            endpoints = span.split('-')
            if len(endpoints) == 1:
                # Single head index
                heads_int.append(int(endpoints[0]))
            elif len(endpoints) == 2:
                # Range specification
                heads_int += torch.arange(
                    int(endpoints[0]), 
                    int(endpoints[1]) + 1
                ).tolist()
            else:
                raise ValueError(f"Invalid head pruning format: {span}")
        
        print(f'[PRUNING] - heads {heads_int} will be pruned')
        config['transformer']['prune_headids'] = heads_int
    else:
        config['transformer']['prune_headids'] = None