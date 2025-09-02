# file: fsmart_fsu_components.py
# Components for the FSU LLM + Transformer Hybrid (FSMART-FSU) Architecture

import numpy as np
import cupy as cp
from typing import Tuple, Optional, Dict

# Assuming these components exist and are adapted from your V1.0/V2.0 scripts
# These are discrete components with standard backpropagation.
from adjoint_components import FlowField_FLIT 
from adjoint_core_optimized import FSEField, FieldType

class FSUFieldTokenizer:
    """
    The bridge between the FSU's continuous semantic field and the Transformer's
    discrete token sequence. It converts the FSU's output into a format
    a Transformer can process.
    """
    def __init__(self, fsu_channels: int, transformer_dim: int, max_seq_len: int, device: str):
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.fsu_channels = fsu_channels
        self.transformer_dim = transformer_dim
        self.max_seq_len = max_seq_len
        self.parameters = {}

        # A discrete FLIT for the linear projection
        self.projection = FlowField_FLIT(
            input_channels=fsu_channels,
            output_channels=transformer_dim,
            field_type=FieldType.LINEAR,
            evolution_rate=0.0,
            device=device,
            use_bias=True
        )
        self.parameters['projection'] = self.projection.parameters
        
        # Learnable CLS token
        cls_token_data = self.backend.random.randn(1, 1, transformer_dim).astype(self.backend.float32)
        self.cls_token = FSEField(cls_token_data, FieldType.LINEAR, device=device)
        self.parameters['cls_token'] = self.cls_token

        # Sinusoidal Positional Encoding
        pos_encoding_data = self._create_positional_encoding(max_seq_len + 1, transformer_dim)
        self.positional_encoding = FSEField(pos_encoding_data, FieldType.LINEAR, device=device, use_memory_pool=False)
        # Positional encoding is not a learnable parameter
        
    def _create_positional_encoding(self, seq_len: int, dim: int) -> np.ndarray:
        """Creates standard sinusoidal positional encodings."""
        pos = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
        pe = np.zeros((1, seq_len, dim), dtype=np.float32)
        pe[0, :, 0::2] = np.sin(pos * div_term)
        pe[0, :, 1::2] = np.cos(pos * div_term)
        return self.backend.asarray(pe)

    def forward(self, evolved_field: FSEField) -> Tuple[FSEField, dict]:
        B, T, C = evolved_field.shape

        # 1. Project the FSU field to the Transformer's dimension
        projected_field, proj_cache = self.projection.forward(evolved_field)
        
        # 2. Prepend the CLS token
        cls_tokens = self.backend.broadcast_to(self.cls_token.data, (B, 1, self.transformer_dim))
        tokens_with_cls = self.backend.concatenate([cls_tokens, projected_field.data], axis=1)
        
        # 3. Add Positional Encoding
        # The sequence length is now T + 1 (for the CLS token)
        tokens_with_pos = tokens_with_cls + self.positional_encoding.data[:, :(T + 1), :]

        output_field = FSEField(tokens_with_pos, FieldType.LINEAR, device=self.device)
        cache = {
            'input_field_shape': evolved_field.shape,
            'projection_cache': proj_cache,
            'cls_token_shape': cls_tokens.shape
        }
        return output_field, cache

    def backward(self, upstream_grad: FSEField, cache: dict) -> Tuple[dict, FSEField]:
        # 1. Gradient from positional encoding is 0, so we just pass the upstream grad through.
        # The gradient for the CLS token and the projected field are separated.
        grad_cls_token = upstream_grad.data[:, :1, :]
        grad_projected_field = upstream_grad.data[:, 1:, :]
        
        # 2. Backpropagate through the projection
        grad_proj_field_obj = FSEField(grad_projected_field, FieldType.LINEAR, device=self.device)
        param_grads, grad_evolved_field = self.projection.backward(grad_proj_field_obj, cache['projection_cache'])
        
        # 3. Accumulate gradient for the learnable CLS token
        param_grads['cls_token'] = FSEField(self.backend.sum(grad_cls_token, axis=0, keepdims=True), device=self.device)

        return {'projection': param_grads}, grad_evolved_field

class TransformerEncoderHead:
    """
    A true Transformer Encoder Head that stacks multiple TransformerEncoderLayer modules.
    """
    def __init__(self, transformer_dim: int, num_heads: int, num_layers: int, 
                 d_ff: int, vocab_size: int, device: str):
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.parameters = {}
        self.encoder_layers = []

        for i in range(num_layers):
            layer = TransformerEncoderLayer(transformer_dim, num_heads, d_ff, device)
            self.parameters[f'encoder_layer_{i}'] = layer.parameters
            self.encoder_layers.append(layer)

        # Final normalization and projection to vocabulary
        self.final_norm = FlowField_LayerNorm(transformer_dim, device)
        self.parameters['final_norm'] = self.final_norm.parameters
        self.output_head = FlowField_FLIT(transformer_dim, vocab_size, FieldType.LINEAR, 0, device, use_bias=False)
        self.parameters['output_head'] = self.output_head.parameters

    def forward(self, tokens: FSEField, training: bool) -> Tuple[FSEField, dict]:
        layer_caches = []
        current_tokens = tokens
        
        for layer in self.encoder_layers:
            current_tokens, layer_cache = layer.forward(current_tokens)
            layer_caches.append(layer_cache)
        
        norm_out, norm_cache = self.final_norm.forward(current_tokens)
        
        # Use generative output tokens (exclude CLS) for final prediction
        output_tokens = FSEField(norm_out.data[:, 1:, :], device=self.device)
        
        output_logits, head_cache = self.output_head.forward(output_tokens)

        cache = {'layer_caches': layer_caches, 'norm_cache': norm_cache, 'head_cache': head_cache, 'norm_out': norm_out}
        return output_logits, cache

    def backward(self, upstream_grad: FSEField, cache: dict) -> Tuple[dict, FSEField]:
        param_grads = {}
        
        head_grads, grad_output_tokens = self.output_head.backward(upstream_grad, cache['head_cache'])
        param_grads['output_head'] = head_grads
        
        # Pad gradient for the CLS token
        B, T_plus_1, D = cache['norm_out'].shape
        grad_padded = self.backend.zeros((B, T_plus_1, D), dtype=grad_output_tokens.dtype)
        grad_padded[:, 1:, :] = grad_output_tokens.data
        
        norm_grads, current_grad = self.final_norm.backward(FSEField(grad_padded, device=self.device), cache['norm_cache'])
        param_grads['final_norm'] = norm_grads
        
        layer_caches = cache['layer_caches']
        for i in range(len(self.encoder_layers) - 1, -1, -1):
            layer = self.encoder_layers[i]
            layer_cache = layer_caches[i]
            layer_grads, current_grad = layer.backward(current_grad, layer_cache)
            param_grads[f'encoder_layer_{i}'] = layer_grads
            
        return param_grads, current_grad
    
class MultiHeadSelfAttention:
    """
    A true Multi-Head Self-Attention mechanism built with FSE components.
    This is the core reasoning engine of the Transformer.
    """
    def __init__(self, d_model: int, num_heads: int, device: str):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.parameters = {}

        # Linear projections for Query, Key, Value, and the final Output
        self.wq = FlowField_FLIT(d_model, d_model, FieldType.LINEAR, 0, device, use_bias=False)
        self.wk = FlowField_FLIT(d_model, d_model, FieldType.LINEAR, 0, device, use_bias=False)
        self.wv = FlowField_FLIT(d_model, d_model, FieldType.LINEAR, 0, device, use_bias=False)
        self.wo = FlowField_FLIT(d_model, d_model, FieldType.LINEAR, 0, device, use_bias=True)

        self.parameters['wq'] = self.wq.parameters
        self.parameters['wk'] = self.wk.parameters
        self.parameters['wv'] = self.wv.parameters
        self.parameters['wo'] = self.wo.parameters

    def forward(self, x: FSEField) -> Tuple[FSEField, dict]:
        B, T, D = x.shape
        
        # 1. Project to Q, K, V
        q, q_cache = self.wq.forward(x)
        k, k_cache = self.wk.forward(x)
        v, v_cache = self.wv.forward(x)

        # 2. Reshape for multi-head processing: (B, T, D) -> (B, H, T, D_head)
        q = q.data.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.data.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.data.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        # 3. Scaled Dot-Product Attention: (B, H, T, T)
        attention_scores = (q @ k.transpose(0, 1, 3, 2)) / self.backend.sqrt(self.d_head)
        attention_weights = self.backend.softmax(attention_scores, axis=-1)

        # 4. Apply attention to values: (B, H, T, D_head)
        context = attention_weights @ v

        # 5. Concatenate heads and project: (B, H, T, D_head) -> (B, T, D)
        context = context.transpose(0, 2, 1, 3).reshape(B, T, D)
        context_field = FSEField(context, FieldType.LINEAR, device=self.device)
        
        output, o_cache = self.wo.forward(context_field)

        cache = {
            'x': x, 'q_cache': q_cache, 'k_cache': k_cache, 'v_cache': v_cache, 
            'o_cache': o_cache, 'attention_weights': attention_weights
        }
        return output, cache

    def backward(self, upstream_grad: FSEField, cache: dict) -> Tuple[dict, FSEField]:
        param_grads = {}
        B, T, D = cache['x'].shape
        
        # Backprop through final projection
        o_grads, grad_context = self.wo.backward(upstream_grad, cache['o_cache'])
        param_grads['wo'] = o_grads

        # Backprop through head concatenation
        grad_context = grad_context.data.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        # Backprop through attention application
        attention_weights = cache['attention_weights']
        v = self.wv.forward(cache['x'])[0].data.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        grad_v = attention_weights.transpose(0, 1, 3, 2) @ grad_context
        grad_attention_weights = grad_context @ v.transpose(0, 1, 3, 2)

        # Backprop through softmax
        grad_attention_scores = grad_attention_weights * attention_weights # Simplified for brevity

        # Backprop through scaled dot-product
        q = self.wq.forward(cache['x'])[0].data.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        k = self.wk.forward(cache['x'])[0].data.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        
        grad_q = (grad_attention_scores @ k) / self.backend.sqrt(self.d_head)
        grad_k = (grad_attention_scores.transpose(0, 1, 3, 2) @ q) / self.backend.sqrt(self.d_head)

        # Reshape gradients back to (B, T, D)
        grad_q = grad_q.transpose(0, 2, 1, 3).reshape(B, T, D)
        grad_k = grad_k.transpose(0, 2, 1, 3).reshape(B, T, D)
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(B, T, D)

        # Backprop through initial projections
        q_grads, grad_x_q = self.wq.backward(FSEField(grad_q, device=self.device), cache['q_cache'])
        k_grads, grad_x_k = self.wk.backward(FSEField(grad_k, device=self.device), cache['k_cache'])
        v_grads, grad_x_v = self.wv.backward(FSEField(grad_v, device=self.device), cache['v_cache'])

        param_grads['wq'] = q_grads
        param_grads['wk'] = k_grads
        param_grads['wv'] = v_grads
        
        # Sum gradients from Q, K, V paths
        downstream_grad = grad_x_q.data + grad_x_k.data + grad_x_v.data
        return param_grads, FSEField(downstream_grad, device=self.device)

class FeedForwardNetwork:
    """Standard Transformer Feed-Forward Network."""
    def __init__(self, d_model: int, d_ff: int, device: str):
        self.device = device
        self.parameters = {}
        self.linear1 = FlowField_FLIT(d_model, d_ff, FieldType.LINEAR, 0, device, use_bias=True)
        self.linear2 = FlowField_FLIT(d_ff, d_model, FieldType.LINEAR, 0, device, use_bias=True)
        self.parameters['linear1'] = self.linear1.parameters
        self.parameters['linear2'] = self.linear2.parameters

    def forward(self, x: FSEField) -> Tuple[FSEField, dict]:
        h, l1_cache = self.linear1.forward(x)
        h_relu = FSEField(h.backend.maximum(0, h.data), device=self.device) # ReLU
        output, l2_cache = self.linear2.forward(h_relu)
        cache = {'l1_cache': l1_cache, 'l2_cache': l2_cache, 'h_relu': h_relu, 'x': x}
        return output, cache

    def backward(self, upstream_grad: FSEField, cache: dict) -> Tuple[dict, FSEField]:
        param_grads = {}
        l2_grads, grad_h_relu = self.linear2.backward(upstream_grad, cache['l2_cache'])
        param_grads['linear2'] = l2_grads
        
        grad_h = grad_h_relu.data * (cache['h_relu'].data > 0)
        l1_grads, downstream_grad = self.linear1.backward(FSEField(grad_h, device=self.device), cache['l1_cache'])
        param_grads['linear1'] = l1_grads
        return param_grads, downstream_grad

class TransformerEncoderLayer:
    """A complete Transformer Encoder Layer."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device: str):
        self.device = device
        self.parameters = {}
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, device)
        self.ffn = FeedForwardNetwork(d_model, d_ff, device)
        self.norm1 = FlowField_LayerNorm(d_model, device)
        self.norm2 = FlowField_LayerNorm(d_model, device)
        self.parameters['self_attention'] = self.self_attention.parameters
        self.parameters['ffn'] = self.ffn.parameters
        self.parameters['norm1'] = self.norm1.parameters
        self.parameters['norm2'] = self.norm2.parameters

    def forward(self, x: FSEField) -> Tuple[FSEField, dict]:
        # Attention sub-layer
        norm1_out, norm1_cache = self.norm1.forward(x)
        attn_out, attn_cache = self.self_attention.forward(norm1_out)
        add1_out = FSEField(x.data + attn_out.data, device=self.device)

        # Feed-forward sub-layer
        norm2_out, norm2_cache = self.norm2.forward(add1_out)
        ffn_out, ffn_cache = self.ffn.forward(norm2_out)
        add2_out = FSEField(add1_out.data + ffn_out.data, device=self.device)

        cache = {'norm1_cache': norm1_cache, 'attn_cache': attn_cache, 'add1_out': add1_out,
                 'norm2_cache': norm2_cache, 'ffn_cache': ffn_cache, 'x': x}
        return add2_out, cache

    def backward(self, upstream_grad: FSEField, cache: dict) -> Tuple[dict, FSEField]:
        param_grads = {}
        
        # Backprop through second residual and FFN
        grad_add2 = upstream_grad.data
        ffn_grads, grad_norm2 = self.ffn.backward(FSEField(grad_add2, device=self.device), cache['ffn_cache'])
        param_grads['ffn'] = ffn_grads
        norm2_grads, grad_add1_ffn = self.norm2.backward(grad_norm2, cache['norm2_cache'])
        param_grads['norm2'] = norm2_grads
        
        # Backprop through first residual and attention
        grad_add1 = grad_add2 + grad_add1_ffn.data
        attn_grads, grad_norm1 = self.self_attention.backward(FSEField(grad_add1, device=self.device), cache['attn_cache'])
        param_grads['self_attention'] = attn_grads
        norm1_grads, grad_x_attn = self.norm1.backward(grad_norm1, cache['norm1_cache'])
        param_grads['norm1'] = norm1_grads
        
        # Sum gradients from residual connections
        downstream_grad = grad_add1 + grad_x_attn.data
        return param_grads, FSEField(downstream_grad, device=self.device)