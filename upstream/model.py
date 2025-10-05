#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio DistilBERT Transformer Model Architecture
==============================================

Core transformer architecture implementation for the Audio DistilBERT framework.
This module implements the complete BERT-style transformer model optimized for
speech representation learning, including masked acoustic modeling and
knowledge distillation capabilities.

Key Components:
- Transformer configuration management
- Multi-head self-attention mechanism
- Position embeddings for audio sequences
- Feed-forward networks with GELU activation
- Masked acoustic modeling prediction heads
- Knowledge distillation support

Author: fanfan-yu
Date: 2025.10.05

Architecture Overview:
    Input: Mel-spectrogram features (BxTxD)
    ↓
    [Position Embedding]
    ↓
    [Multi-Head Self-Attention]
    ↓
    [Feed-Forward Network]
    ↓
    [Masked Prediction Head]
    ↓
    Output: Reconstructed/Enhanced features
"""

import math
import copy

from torch import nn
import torch
from utils import prune_linear_layer, gelu

class TransformerConfig(object):
    """
    Configuration class for Transformer model architecture.
    
    This class encapsulates all hyperparameters and architectural choices
    for the Audio DistilBERT transformer model, enabling reproducible
    experiments and easy configuration management.
    
    Attributes:
        downsample_rate (int): Frame stacking rate for sequence compression
        hidden_size (int): Transformer hidden dimension
        num_hidden_layers (int): Number of transformer layers
        num_attention_heads (int): Number of attention heads per layer
        hidden_act (str): Activation function type ('gelu')
        intermediate_size (int): Feed-forward network hidden size
        hidden_dropout_prob (float): Dropout probability for hidden layers
        attention_probs_dropout_prob (float): Dropout for attention weights
        initializer_range (float): Xavier initialization standard deviation
        layer_norm_eps (float): Layer normalization epsilon for stability
        share_layer (bool): Whether to share layer weights across depths
        pre_layer_norm (bool): Apply layer norm before or after attention
        
    Example:
        >>> config = TransformerConfig({
        ...     'hidden_size': 768,
        ...     'num_hidden_layers': 12,
        ...     'num_attention_heads': 12,
        ... })
    """
    
    def __init__(self, config):
        """Initialize transformer configuration from nested dictionary."""
        self.downsample_rate = config['transformer']['downsample_rate']
        self.hidden_size = config['transformer']['hidden_size']
        self.num_hidden_layers = config['transformer']['num_hidden_layers']
        self.num_attention_heads = config['transformer']['num_attention_heads']
        self.hidden_act = config['transformer']['hidden_act']
        self.intermediate_size = config['transformer']['intermediate_size']
        self.hidden_dropout_prob = config['transformer']['hidden_dropout_prob']
        self.attention_probs_dropout_prob = config['transformer']['attention_probs_dropout_prob']
        self.initializer_range = config['transformer']['initializer_range']
        self.layer_norm_eps = float(config['transformer']['layer_norm_eps'])
        self.share_layer = bool(config['transformer']['share_layer']) if 'share_layer' in config['transformer'] else False
        self.pre_layer_norm = bool(config['transformer']['pre_layer_norm']) if 'pre_layer_norm' in config['transformer'] else False


class TransformerInitModel(nn.Module):
    """
    Base transformer model with weight initialization utilities.
    
    This abstract class provides common weight initialization strategies
    used throughout the Audio DistilBERT architecture, ensuring consistent
    initialization across different model variants.
    
    Methods:
        init_Transformer_weights: Initialize transformer model weights
    """
    
    def __init__(self, config):
        super(TransformerInitModel, self).__init__()
        self.config = config

    def init_Transformer_weights(self, module):
        """
        Initialize transformer weights using Xavier/He initialization.
        
        This method applies appropriate initialization to different module types:
        - Linear layers: Xavier normal initialization
        - Embedding layers: Xavier normal initialization
        - LayerNorm: Zero bias and unit weight
        
        Args:
            module (nn.Module): PyTorch module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, TransformerLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class TransformerLayerNorm(nn.Module):
    """
    Layer normalization with learnable parameters.
    
    This implementation follows the standard layer normalization
    followed by element-wise affine transformation with learnable
    parameters, critical for transformer training stability.
    
    Args:
        hidden_size (int): Hidden dimension size
        eps (float): Small epsilon for numerical stability
    """
    
    def __init__(self, hidden_size, eps=1e-12):
        super(TransformerLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        """
        Apply layer normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., hidden_size)
            
        Returns:
            torch.Tensor: Normalized tensor with same shape as input
        """
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class TransformerSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for audio sequences.
    
    This implementation provides the core attention mechanism used in Audio DistilBERT,
    optimized for handling variable-length audio sequences with proper masking support.
    
    Architecture:
    - Linear projections for Q, K, V from same input
    - Multi-head attention with configurable heads
    - Dropout for regularization
    - Optional head masking for ablation studies
    
    Args:
        config (TransformerConfig): Model configuration object
        
    Input Shape:
        - hidden_states: (batch_size, seq_len, hidden_size)
        - attention_mask: (batch_size, seq_len)
        - head_mask: (num_heads,) optional
    """
    
    def __init__(self, config):
        super(TransformerSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.multihead_output = None

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear projections for Q, K, V
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        Reshape tensor for multi-head attention computation.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, all_head_size)
            
        Returns:
            torch.Tensor: Reshaped tensor (batch_size, num_heads, seq_len, head_size)
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        Forward pass for multi-head self-attention.
        
        Args:
            hidden_states (torch.Tensor): Input sequence (batch_size, seq_len, hidden_size)
            attention_mask (torch.Tensor): Mask for padding (batch_size, 1, 1, seq_len)
            head_mask (torch.Tensor, optional): Mask for attention heads (num_heads,)
            
        Returns:
            torch.Tensor: Attention output (batch_size, seq_len, hidden_size)
        """
        # Project to Q, K, V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Reshape for multi-head computation
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        attention_scores = attention_scores + attention_mask

        # Convert to probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # Apply dropout
        attention_probs = self.dropout(attention_probs)

        # Mask specific heads if provided
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to original form
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class TransformerSelfOutput(nn.Module):
    """
    Output projection and residual connection for self-attention.
    
    This module handles the final linear projection after attention
    and adds residual connection with layer normalization for training stability.
    
    Args:
        config (TransformerConfig): Model configuration
    """
    
    def __init__(self, config):
        super(TransformerSelfOutput, self).__init__()
        self.pre_layer_norm = config.pre_layer_norm
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = TransformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        """
        Forward pass with residual connection.
        
        Args:
            hidden_states (torch.Tensor): Attention output (batch_size, seq_len, hidden_size)
            input_tensor (torch.Tensor): Residual input (batch_size, seq_len, hidden_size)
            
        Returns:
            torch.Tensor: Normalized output with residual connection
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TransformerAttention(nn.Module):
    """
    Complete transformer attention block with pruning support.
    
    This module combines self-attention with output projection and includes
    functionality to prune attention heads for model compression.
    
    Args:
        config (TransformerConfig): Model configuration
        
    Methods:
        prune_heads: Remove specific attention heads from the model
    """
    
    def __init__(self, config):
        super(TransformerAttention, self).__init__()
        self.pre_layer_norm = config.pre_layer_norm
        self.self = TransformerSelfAttention(config)
        self.output = TransformerSelfOutput(config)
        if self.pre_layer_norm:
            self.LayerNorm = self.output.LayerNorm

    def prune_heads(self, heads):
        """
        Prune specific attention heads from the model.
        
        This method reduces model size by removing specified attention heads
        while maintaining the ability to perform inference.
        
        Args:
            heads (list): List of head indices to prune (0-indexed)
            
        Example:
            >>> attention.prune_heads([0, 2, 5])  # Remove heads 0, 2, and 5
        """
        if len(heads) == 0:
            return
        
        # Create mask for remaining heads
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        
        # Update hyperparameters
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask, head_mask=None):
        """
        Forward pass for attention block.
        
        Args:
            input_tensor (torch.Tensor): Input sequence (batch_size, seq_len, hidden_size)
            attention_mask (torch.Tensor): Padding mask (batch_size, 1, 1, seq_len)
            head_mask (torch.Tensor, optional): Head mask (num_heads,)
            
        Returns:
            torch.Tensor: Attention output with residual connection
        """
        attention_output = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(attention_output, input_tensor)
        return attention_output


class TransformerIntermediate(nn.Module):
    """
    Feed-forward network intermediate layer.
    
    This module implements the feed-forward network (FFN) component
    of the transformer architecture with GELU activation.
    
    Args:
        config (TransformerConfig): Model configuration
    """
    
    def __init__(self, config):
        super(TransformerIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        """
        Forward pass through FFN intermediate layer.
        
        Args:
            hidden_states (torch.Tensor): Input (batch_size, seq_len, hidden_size)
            
        Returns:
            torch.Tensor: Intermediate representation (batch_size, seq_len, intermediate_size)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    """
    Complete transformer layer with self-attention and FFN.
    
    A single transformer layer consisting of:
    1. Multi-head self-attention with residual connection
    2. Feed-forward network with residual connection
    3. Layer normalization
    
    Args:
        config (TransformerConfig): Model configuration
    """
    
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.pre_layer_norm = config.pre_layer_norm
        self.attention = TransformerAttention(config)
        self.intermediate = TransformerIntermediate(config)
        self.output = TransformerOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        Forward pass through transformer layer.
        
        Args:
            hidden_states (torch.Tensor): Input (batch_size, seq_len, hidden_size)
            attention_mask (torch.Tensor): Padding mask (batch_size, 1, 1, seq_len)
            head_mask (torch.Tensor, optional): Head mask (num_heads,)
            
        Returns:
            torch.Tensor: Layer output (batch_size, seq_len, hidden_size)
        """
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class TransformerOutput(nn.Module):
    """
    Output layer for transformer FFN with residual connection.
    
    This module handles the final projection of the feed-forward network
    and adds residual connection with layer normalization.
    
    Args:
        config (TransformerConfig): Model configuration
    """
    
    def __init__(self, config):
        super(TransformerOutput, self).__init__()
        self.pre_layer_norm = config.pre_layer_norm
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = TransformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        """
        Forward pass with residual connection.
        
        Args:
            hidden_states (torch.Tensor): FFN output (batch_size, seq_len, intermediate_size)
            input_tensor (torch.Tensor): Residual input (batch_size, seq_len, hidden_size)
            
        Returns:
            torch.Tensor: Final output (batch_size, seq_len, hidden_size)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        if not self.pre_layer_norm:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class TransformerSelfAttention(nn.Module):
    def __init__(self, config):
        super(TransformerSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.multihead_output = None

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # each mixed layer: (batch_size, seqlen, head_num * head_dim)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # each layer: (batch_size, head_num, seqlen, head_dim)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in TransformerModel forward() function)
        attention_scores = attention_scores + attention_mask
        # attention_scores: (batch_size, head_num, seqlen, seqlen)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer: (batch_size, head_num, seqlen, head_dim)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class TransformerSelfOutput(nn.Module):
    def __init__(self, config):
        super(TransformerSelfOutput, self).__init__()
        self.pre_layer_norm = config.pre_layer_norm
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = TransformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class TransformerAttention(nn.Module):
    def __init__(self, config):
        super(TransformerAttention, self).__init__()
        self.pre_layer_norm = config.pre_layer_norm
        self.self = TransformerSelfAttention(config)
        self.output = TransformerSelfOutput(config)
        if self.pre_layer_norm:
            self.LayerNorm = self.output.LayerNorm

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask, head_mask=None):
        # SelfAttention -> SelfOutput (residual + LayerNorm)
        self_output = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class TransformerIntermediate(nn.Module):
    def __init__(self, config):
        super(TransformerIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.pre_layer_norm = config.pre_layer_norm
        self.attention = TransformerAttention(config)
        self.intermediate = TransformerIntermediate(config)
        self.output = TransformerOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        if self.output_attentions:
            attentions, attention_output = attention_output
        # Intermediate -> Output (residual + LayerNorm)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class TransformerOutput(nn.Module):
    def __init__(self, config):
        super(TransformerOutput, self).__init__()
        self.pre_layer_norm = config.pre_layer_norm
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = TransformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps) # layer_norm for FFN

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        if not self.pre_layer_norm:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class TransformerInputRepresentations(nn.Module):
    """
    Input representation construction from spectrogram features.
    
    This module converts raw mel-spectrogram features into transformer-ready
    representations by applying linear projection, adding positional encodings,
    and applying layer normalization and dropout.
    
    Args:
        config (TransformerConfig): Model configuration
        input_dim (int): Input feature dimension (e.g., 80 for mel-spectrogram)
        
    Architecture:
    - Linear projection from input_dim*downsample_rate to hidden_size
    - Position encoding integration
    - Layer normalization and dropout
    
    Input Shape:
        - spec: (batch_size, seq_len, input_dim)
        - pos_enc: (seq_len, hidden_size)
        
    Output Shape:
        - (batch_size, seq_len, hidden_size)
    """
    
    def __init__(self, config, input_dim):
        super(TransformerInputRepresentations, self).__init__()
        self.hidden_size = config.hidden_size
        
        # Linear projection for feature dimension alignment
        self.spec_transform = nn.Linear(input_dim * config.downsample_rate, config.hidden_size)
        
        # Layer normalization and dropout
        self.LayerNorm = TransformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, spec, pos_enc):
        """
        Build input representations from spectrogram and position encodings.
        
        Args:
            spec (torch.Tensor): Input spectrogram (batch_size, seq_len, input_dim*dr)
            pos_enc (torch.Tensor): Position encodings (seq_len, hidden_size)
            
        Returns:
            torch.Tensor: Transformer input representations
        """
        spec_transformed = self.spec_transform(spec)
        
        # Add position encodings
        input_representations = spec_transformed + pos_enc
        
        # Apply normalization and dropout
        input_representations = self.LayerNorm(input_representations)
        input_representations = self.dropout(input_representations)
        return input_representations


class TransformerEncoder(nn.Module):
    """
    Stack of transformer layers for sequence encoding.
    
    This module implements the core transformer encoder consisting
    of multiple stacked transformer layers with optional weight sharing.
    
    Args:
        config (TransformerConfig): Model configuration
        
    Output:
        List of layer outputs, one per layer
    """
    
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(config)
        
        # Create layer stack with optional weight sharing
        if config.share_layer:
            self.layer = nn.ModuleList([layer] * config.num_hidden_layers)
        else:
            self.layer = nn.ModuleList([copy.deepcopy(layer) 
                                      for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        Forward pass through transformer encoder.
        
        Args:
            hidden_states (torch.Tensor): Input embeddings (batch_size, seq_len, hidden_size)
            attention_mask (torch.Tensor): Padding mask (batch_size, 1, 1, seq_len)
            head_mask (list, optional): Mask for each layer's heads
            
        Returns:
            list: List of layer outputs [final_layer_output]
        """
        all_encoder_layers = []
        
        # Process through all transformer layers
        for i, layer_module in enumerate(self.layer):
            if head_mask is not None:
                layer_head_mask = head_mask[i]
            else:
                layer_head_mask = None
                
            hidden_states = layer_module(hidden_states, attention_mask, layer_head_mask)
            
        all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class TransformerModel(TransformerInitModel):
    """
    Complete transformer model for audio sequence processing.
    
    This is the main transformer backbone that processes audio sequences
    through embedding, position encoding, and transformer layers.
    
    Args:
        config (TransformerConfig): Model configuration
        input_dim (int): Input feature dimension
        output_attentions (bool): Whether to output attention weights
        with_input_module (bool): Whether to include input embeddings
    """
    
    def __init__(self, config, input_dim, output_attentions=False, with_input_module=True):
        super(TransformerModel, self).__init__(config)
        self.with_input_module = with_input_module
        
        if self.with_input_module:
            self.input_representations = TransformerInputRepresentations(config, input_dim)
        self.encoder = TransformerEncoder(config)
        
        # Initialize weights
        self.apply(self.init_Transformer_weights)

    def prune_heads(self, heads_to_prune):
        """
        Prune attention heads from specific layers.
        
        Args:
            heads_to_prune (dict): Dictionary mapping layer index to list of heads
                                  Example: {0: [0, 1], 2: [5, 6, 7]}
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_multihead_outputs(self):
        """
        Extract intermediate outputs from all attention heads.
        
        Returns:
            list: List of multihead attention outputs from each layer
        """
        return [layer.attention.self.multihead_output 
                for layer in self.encoder.layer]

    def forward(self, spec_input, pos_enc=None, attention_mask=None, head_mask=None):
        """
        Forward pass through complete transformer model.
        
        Args:
            spec_input (torch.Tensor): Input spectrogram (batch_size, seq_len, input_dim)
            pos_enc (torch.Tensor, optional): Position encodings
            attention_mask (torch.Tensor, optional): Attention mask
            head_mask (list, optional): Per-layer head masks
            
        Returns:
            torch.Tensor: Final encoded representations (batch_size, seq_len, hidden_size)
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(spec_input)

        # Build 3D attention mask from 2D tensor
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Convert mask to proper format
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Handle head masking
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Build input representations
        if self.with_input_module:
            input_representations = self.input_representations(spec_input, pos_enc)
        else:
            input_representations = spec_input
            
        # Process through transformer layers
        encoded_layers = self.encoder(
            input_representations, extended_attention_mask, head_mask)
        
        return encoded_layers[-1]


class TransformerSpecPredictionHead(nn.Module):
    """
    Prediction head for masked acoustic modeling.
    
    This module reconstructs the original spectrogram from transformer
    hidden states using a simple feed-forward network with GELU activation.
    
    Args:
        config (TransformerConfig): Model configuration
        output_dim (int): Output feature dimension
        input_dim (int, optional): Override input dimension, defaults to None
    """
    
    def __init__(self, config, output_dim, input_dim=None):
        super(TransformerSpecPredictionHead, self).__init__()
        self.output_dim = output_dim
        
        # Optional input dimension override
        if input_dim is None:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.dense = nn.Linear(input_dim, config.hidden_size)
            
        self.transform_act_fn = gelu
        self.LayerNorm = TransformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Final projection to output dimension
        self.output = nn.Linear(config.hidden_size, 
                              self.output_dim * config.downsample_rate)

    def forward(self, hidden_states):
        """
        Predict masked spectrogram features.
        
        Args:
            hidden_states (torch.Tensor): Transformer output (batch_size, seq_len, hidden_size)
            
        Returns:
            tuple: (predicted_spectrogram, intermediate_representation)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        linear_output = self.output(hidden_states)
        return linear_output, hidden_states


class UpstreamModel(TransformerInitModel):
    """
    Complete upstream model for speech representation learning.
    
    This is the full Audio DistilBERT model that combines the transformer
    encoder with a masked acoustic modeling prediction head for
    self-supervised learning.
    
    Args:
        config (TransformerConfig): Model configuration
        input_dim (int): Input feature dimension
        output_dim (int, optional): Output dimension, defaults to input_dim
    """
    
    def __init__(self, config, input_dim, output_dim):
        super(UpstreamModel, self).__init__(config)
        
        # Core transformer encoder
        self.Transformer = TransformerModel(config, input_dim)
        
        # Prediction head for masked acoustic modeling
        self.SpecHead = TransformerSpecPredictionHead(
            config, 
            output_dim if output_dim is not None else input_dim
        )
        
        # Loss function for reconstruction
        self.loss = nn.L1Loss()

    def forward(self, spec_input, pos_enc, mask_label=None, 
                attention_mask=None, spec_label=None, head_mask=None):
        """
        Forward pass through complete upstream model.
        
        Args:
            spec_input (torch.Tensor): Input spectrogram (batch_size, seq_len, input_dim)
            pos_enc (torch.Tensor): Position encodings
            mask_label (torch.Tensor, optional): Mask indicators
            attention_mask (torch.Tensor, optional): Attention mask
            spec_label (torch.Tensor, optional): Ground truth for loss computation
            head_mask (list, optional): Head masks for attention
            
        Returns:
            Union[tuple, torch.Tensor]: 
                - If labels provided: (loss, predictions)
                - Else: predictions
                
        Masking:
        The loss is only computed on masked positions to encourage
        learning of meaningful representations.
        """
        # Process through transformer
        outputs = self.Transformer(
            spec_input, pos_enc, attention_mask, head_mask=head_mask)
            
        sequence_output = outputs
        
        # Predict masked spectrogram
        pred_spec, pred_state = self.SpecHead(sequence_output)

        # Compute loss if labels provided
        if spec_label is not None and mask_label is not None:
            assert mask_label.sum() > 0, \
                'Without any masking, loss might go NaN. Modify your data preprocessing'
            
            masked_spec_loss = self.loss(
                pred_spec.masked_select(mask_label), 
                spec_label.masked_select(mask_label)
            )
            return masked_spec_loss, pred_spec
            
        return pred_spec, pred_state
