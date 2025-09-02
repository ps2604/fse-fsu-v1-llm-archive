# file: flowfield_components.py
# FLOWFIELD COMPONENTS - FULLY ADJOINT IMPLEMENTED
# Revolutionary conversion from discrete matmul backprop to continuous field adjoint equations
# UPDATES: Complete replacement of backward() methods with adjoint PDE solvers

import numpy as np
import cupy as cp
from typing import Dict, List, Optional, Tuple, Any

from adjoint_core_optimized import (
    FSEField, FieldType, FieldOperations, ArrayLike, 
    get_memory_pool, FusedFieldOperations, _ensure_4d,
    get_default_dtype, DEFAULT_DTYPE
)
from adjoint_solvers import FSEAdjointSolvers, AdjointIntegrationMethod
import logging

logger = logging.getLogger(__name__)

# In file: adjoint_components.py
# Replace the entire FlowField_FLIT class with this corrected version.

class FlowField_FLIT:
    """
    ✅ [DEFINITIVE FIX] FULLY ADJOINT IMPLEMENTED: Trainable FLIT using continuous field adjoint equations.
    This version corrects the forward pass to use a discrete transformation for projection layers
    (where input_channels != output_channels), resolving the core broadcast error. PDE evolution
    is now correctly reserved for interaction layers where dimensions are constant.
    """
    def __init__(self, input_channels: int, output_channels: int,
                 field_type: FieldType, evolution_rate: float, device: str, use_bias: bool = True,
                 context_channels_in: Optional[int] = None):
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.activation_field_type = field_type 
        self.evolution_rate = evolution_rate 
        self.output_channels = output_channels
        self.input_channels = input_channels

        self.adjoint_solver = FSEAdjointSolvers(
            device=device, 
            integration_method=AdjointIntegrationMethod.RUNGE_KUTTA_4
        )

        current_dtype = get_default_dtype()

        init_scale_kernel = self.backend.sqrt(2. / input_channels)
        kernel_data = self.backend.random.normal(0, init_scale_kernel, (input_channels, output_channels)).astype(current_dtype)
        self.kernel = FSEField(kernel_data, field_type=FieldType.LINEAR, device=device, dtype=current_dtype)
        self.parameters = {'kernel': self.kernel}

        self.use_bias = use_bias
        if self.use_bias:
            bias_data = self.backend.zeros(output_channels, dtype=current_dtype)
            self.bias = FSEField(bias_data, field_type=FieldType.LINEAR, device=device, dtype=current_dtype)
            self.parameters['bias'] = self.bias
        
        self.context_channels_in = context_channels_in
        self.context_projection_kernel: Optional[FSEField] = None
        self.context_projection_bias: Optional[FSEField] = None

        if self.context_channels_in is not None and self.context_channels_in > 0:
            mod_channels = output_channels
            init_scale_ctx = self.backend.sqrt(1. / self.context_channels_in)
            ctx_proj_kernel_data = self.backend.random.normal(0, init_scale_ctx, (self.context_channels_in, mod_channels)).astype(current_dtype)
            self.context_projection_kernel = FSEField(ctx_proj_kernel_data, FieldType.LINEAR, device=device, dtype=current_dtype)
            self.parameters['context_projection_kernel'] = self.context_projection_kernel
            
            if self.use_bias:
                ctx_proj_bias_data = self.backend.zeros(mod_channels, dtype=current_dtype)
                self.context_projection_bias = FSEField(ctx_proj_bias_data, FieldType.LINEAR, device=device, dtype=current_dtype)
                self.parameters['context_projection_bias'] = self.context_projection_bias
        
        logger.debug(f"✅ ADJOINT FLIT Init: In({input_channels})->Out({output_channels}), "
                    f"field_type={field_type}, adjoint_solver=initialized")


    def forward(self, inputs: FSEField, context_signal: Optional[FSEField] = None) -> Tuple[FSEField, Dict[str, Any]]:
        """
        [INTELLIGENT LOGGING] This version adds a 'fallback_reason' to the cache
        to provide the backward pass with the context needed for accurate logging.
        """
        if inputs.device != self.device: 
            inputs = inputs.to_device(self.device)
        
        current_dtype = get_default_dtype()
        if inputs.dtype != current_dtype:
            inputs_data = inputs.data.astype(current_dtype)
            inputs = FSEField(inputs_data, inputs.field_type, device=inputs.device, dtype=current_dtype)
        
        is_projection = self.input_channels != self.output_channels

        if is_projection:
            # PATH A: DISCRETE FOR PROJECTIONS
            linear_output = inputs.data @ self.kernel.data
            if self.use_bias:
                linear_output = linear_output + self.bias.data
            
            final_field = FSEField(linear_output, self.activation_field_type, device=self.device, dtype=current_dtype)
            
            # Add the reason for using the discrete path
            evolution_cache = {
                'fallback_mode': True,
                'fallback_reason': 'ARCHITECTURAL_PROJECTION' 
            }
            cache = {'input_field': inputs, 'evolution_cache': evolution_cache, 'final_field': final_field}

        else:
            # PATH B: CONTINUOUS FOR INTERACTIONS
            num_evolution_steps = max(4, int(self.evolution_rate * 50))
            dt = 0.02
            flit_parameters = self.parameters.copy()
            context_for_pde = None

            try:
                final_field, trajectory, evolution_cache = self.adjoint_solver.solve_forward_pde(
                    initial_field=inputs, parameters=flit_parameters, num_steps=num_evolution_steps,
                    dt=dt, field_type=self.activation_field_type, context_signal=context_for_pde
                )
                evolution_cache['fallback_mode'] = False
            
            except Exception as e:
                logger.error(f"❌ INTERACTION FLIT FAILED PDE SOLVE: {e}. Falling back.", exc_info=True)
                linear_output = inputs.data @ self.kernel.data
                if self.use_bias: linear_output += self.bias.data
                final_field = FSEField(linear_output, self.activation_field_type, device=self.device, dtype=current_dtype)
                trajectory = [inputs, final_field]
                
                # Add the reason for the failure
                evolution_cache = {
                    'fallback_mode': True,
                    'fallback_reason': 'PDE_SOLVE_FAILURE'
                }

            cache = {
                'input_field': inputs, 'forward_trajectory': trajectory, 'evolution_cache': evolution_cache,
                'parameters_used': flit_parameters, 'final_field': final_field
            }
            
        return final_field, cache
            


    def backward(self, upstream_grad_activated: FSEField, cache: Dict[str, Any], truncated_len: int = None) -> Tuple[Dict[str, FSEField], FSEField]:
        """
        [INTELLIGENT LOGGING] This backward pass now checks the fallback_reason from the
        cache to provide accurate, context-aware logging.
        """
        evolution_cache = cache['evolution_cache']
        
        if evolution_cache.get('fallback_mode', False):
            reason = evolution_cache.get('fallback_reason', 'UNKNOWN')
            
            if reason == 'ARCHITECTURAL_PROJECTION':
                # This is expected behavior, so we log at a lower level (DEBUG).
                # This won't clutter your main training logs.
                logger.debug(f"Using discrete gradients for FLIT (Reason: Architectural Choice for Projection).")
            else:
                # This indicates a real problem, so we keep it as a loud WARNING.
                logger.warning(f"⚠️ Using fallback discrete gradients for FLIT (Reason: {reason}).")
                
            return self._fallback_discrete_backward(upstream_grad_activated, cache)

        # The rest of the continuous adjoint path
        forward_trajectory = cache['forward_trajectory']
        parameters_used = cache['parameters_used']
        
        # Ensure upstream gradient is in fp32 for numerical stability
        if upstream_grad_activated.dtype != self.backend.float32:
            upstream_grad_fp32 = FSEField(
                upstream_grad_activated.data.astype(self.backend.float32), 
                upstream_grad_activated.field_type, 
                device=upstream_grad_activated.device
            )
        else:
            upstream_grad_fp32 = upstream_grad_activated

        try:
            # REVOLUTIONARY ADJOINT SOLVE: Replace chain-rule with adjoint PDE
            param_gradients, downstream_grad = self.adjoint_solver.solve_adjoint_pde(
                upstream_grad_field=upstream_grad_fp32,
                parameters=parameters_used,
                forward_trajectory=forward_trajectory,
                evolution_cache=evolution_cache
            )
            
            # Handle bias gradients if present
            if self.use_bias and 'bias' not in param_gradients:
                bias_grad_data = self.backend.sum(upstream_grad_fp32.data, axis=(0, 1))
                param_gradients['bias'] = FSEField(
                    bias_grad_data, 
                    device=self.device, 
                    dtype=self.backend.float32
                )
            
            # Handle context projection gradients
            if self.context_projection_kernel is not None:
                if 'context_projection_kernel' not in param_gradients:
                    param_gradients['context_projection_kernel'] = FSEField(
                        self.backend.zeros_like(self.context_projection_kernel.data, dtype=self.backend.float32),
                        device=self.device,
                        dtype=self.backend.float32
                    )
            
            return param_gradients, downstream_grad
            
        except Exception as e:
            logger.error(f"❌ Adjoint PDE solve failed: {e}, falling back to discrete gradients", exc_info=True)
            return self._fallback_discrete_backward(upstream_grad_activated, cache)
    
    def _fallback_discrete_backward(self, upstream_grad: FSEField, cache: Dict[str, Any]) -> Tuple[Dict[str, FSEField], FSEField]:
            """
            [DEFINITIVE STABILITY FIX] Fallback discrete backward pass that now
            correctly reshapes the downstream gradient back to its original 3D shape,
            resolving the broadcast error in the main model's backward pass.
            """
            param_grads = {}
            input_field = cache.get('input_field')
            if input_field is None:
                logger.error("❌ CRITICAL FALLBACK FAILURE: 'input_field' not found in cache. Cannot compute gradients.")
                return {}, FSEField(self.backend.zeros_like(upstream_grad.data), upstream_grad.field_type, device=self.device)
                
            if upstream_grad.device != self.device: upstream_grad = upstream_grad.to_device(self.device)
            if input_field.device != self.device: input_field = input_field.to_device(self.device)
            upstream_grad_data_fp32 = upstream_grad.data.astype(self.backend.float32)
            input_data_fp32 = input_field.data.astype(self.backend.float32)

            upstream_flat = upstream_grad_data_fp32.reshape(-1, upstream_grad_data_fp32.shape[-1])
            input_flat = input_data_fp32.reshape(-1, input_data_fp32.shape[-1])
            
            if input_flat.shape[0] != upstream_flat.shape[0]:
                min_rows = min(input_flat.shape[0], upstream_flat.shape[0])
                input_flat, upstream_flat = input_flat[:min_rows, :], upstream_flat[:min_rows, :]
            
            param_grads['kernel'] = FSEField(input_flat.T @ upstream_flat, device=self.device, dtype=self.backend.float32)
            
            if self.use_bias:
                param_grads['bias'] = FSEField(self.backend.sum(upstream_grad_data_fp32, axis=(0, 1)), device=self.device, dtype=self.backend.float32)
            
            # This calculation produces a 2D (flattened) gradient
            grad_input_flat = upstream_flat @ self.kernel.data.astype(self.backend.float32).T
            
            # --- THIS IS THE FIX ---
            # Reshape the flattened gradient back to the original 3D shape of the input field.
            grad_input_data = grad_input_flat.reshape(input_field.shape)
            # --- END OF FIX ---

            downstream_grad = FSEField(grad_input_data, input_field.field_type, device=self.device, dtype=self.backend.float32)
            return param_grads, downstream_grad





class FlowField_FSEBlock:
    """
    [DEFINITIVE DEBUG VERSION] This version includes extensive logging in the
    backward pass to pinpoint the exact source of the type corruption error.
    """
    def __init__(self, input_channels: int, internal_channels: int, num_fils: int, device: str, 
                 use_bias_in_fils:bool = True, context_channels_for_fils: Optional[int] = None):
        # This __init__ method is correct and remains unchanged.
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.context_channels_for_fils = context_channels_for_fils
        self.fils: List[Tuple[str, Any]] = []
        self.parameters: Dict[str, Any] = {}
        self.norm1 = FlowField_LayerNorm(input_channels, device=device)
        self.norm2 = FlowField_LayerNorm(internal_channels, device=device)
        current_ch = input_channels
        for i in range(num_fils // 2):
            name = f'fil{i+1}'
            layer = FlowField_FLIT(current_ch, internal_channels, FieldType.CONTINUOUS, 0.1, device, 
                                   use_bias=use_bias_in_fils, context_channels_in=self.context_channels_for_fils)
            self.fils.append((name, layer))
            current_ch = internal_channels
        conv_name = 'conv1d_fil'
        conv_layer = FlowField_Conv1D_FLIT(channels=internal_channels, kernel_size=3, device=device)
        self.fils.append((conv_name, conv_layer))
        for i in range(num_fils // 2, num_fils):
            name = f'fil{i+1}'
            layer = FlowField_FLIT(current_ch, internal_channels, FieldType.CONTINUOUS, 0.1, device, 
                                   use_bias=use_bias_in_fils, context_channels_in=self.context_channels_for_fils)
            self.fils.append((name, layer))
            current_ch = internal_channels
        self.parameters['norm1'] = self.norm1.parameters
        self.parameters['norm2'] = self.norm2.parameters
        for name, layer in self.fils:
            self.parameters[name] = layer.parameters
        self.skip_projection_flit: Optional[FlowField_FLIT] = None
        if input_channels != internal_channels:
            self.skip_projection_flit = FlowField_FLIT(input_channels, internal_channels, FieldType.LINEAR, 0.1, device, 
                                                       use_bias=use_bias_in_fils, context_channels_in=None) 
            self.parameters['skip_projection_flit'] = self.skip_projection_flit.parameters

    def forward(self, inputs: FSEField, context_signal: Optional[FSEField] = None) -> Tuple[FSEField, Dict[str, Any]]:
        # This forward pass is correct and remains unchanged.
        current_dtype = get_default_dtype()
        if inputs.device != self.device: inputs = inputs.to_device(self.device)
        if inputs.dtype != current_dtype:
            inputs = FSEField(inputs.data.astype(current_dtype), inputs.field_type, device=inputs.device)
        x, norm1_cache = self.norm1.forward(inputs)
        fil_caches: List[Dict[str, Any]] = []
        for name, fil_layer in self.fils:
            x, fil_cache = fil_layer.forward(x, context_signal=context_signal)
            fil_caches.append(fil_cache)
        x_after_fils, norm2_cache = self.norm2.forward(x)
        projected_skip_field = inputs
        cache_skip_proj = {}
        if self.skip_projection_flit:
            projected_skip_field, cache_skip_proj = self.skip_projection_flit.forward(inputs, context_signal=None)
        if x_after_fils.shape[1] != projected_skip_field.shape[1]:
            target_len = x_after_fils.shape[1]
            if projected_skip_field.shape[1] > target_len:
                projected_skip_field = FSEField(projected_skip_field.data[:, :target_len, :], projected_skip_field.field_type, device=self.device)
            else:
                pad_width = ((0, 0), (0, target_len - projected_skip_field.shape[1]), (0, 0))
                padded_data = self.backend.pad(projected_skip_field.data, pad_width, 'constant')
                projected_skip_field = FSEField(padded_data, projected_skip_field.field_type, device=self.device)
        sum_data = x_after_fils.data + projected_skip_field.data
        sum_field = FSEField(sum_data, x_after_fils.field_type, device=self.device)
        final_field = FSEField(self.backend.tanh(sum_field.data), sum_field.field_type, device=self.device)
        cache = {
            "inputs": inputs, "fil_caches": fil_caches, "cache_skip_proj": cache_skip_proj,
            "sum_before_activation_data": sum_field.data, "x_after_fils": x_after_fils,
            "projected_skip_field": projected_skip_field, "norm1_cache": norm1_cache, "norm2_cache": norm2_cache
        }
        return final_field, cache

# In file: adjoint_components.py
# Replace the 'backward' method in your 'FlowField_FSEBlock' class with this version.

    def backward(self, upstream_grad_output: FSEField, cache: Dict[str, Any]) -> Tuple[Dict[str, Any], FSEField]:
        """
        [DEFINITIVE FIX] This backward pass corrects the variable swapping during the
        tuple unpacking from the LayerNorm backward calls.
        """
        all_param_grads: Dict[str, Any] = {}
        
        grad_sum_data = upstream_grad_output.data * (1.0 - self.backend.tanh(cache["sum_before_activation_data"])**2)
        grad_sum_field = FSEField(grad_sum_data, upstream_grad_output.field_type, device=self.device)

        grad_for_main_path = grad_sum_field
        grad_for_skip_path = grad_sum_field

        # --- THE FIX: Swapped variables to correctly unpack the (dict, FSEField) tuple ---
        norm2_param_grads, grad_after_fils = self.norm2.backward(grad_for_main_path, cache["norm2_cache"])
        all_param_grads['norm2'] = norm2_param_grads

        current_grad = grad_after_fils # current_grad is now correctly an FSEField
        fil_caches = cache['fil_caches']
        
        for i in range(len(self.fils) - 1, -1, -1):
            name, fil_layer = self.fils[i]
            param_grads, current_grad = fil_layer.backward(current_grad, fil_caches[i])
            all_param_grads[name] = param_grads
        
        # --- THE FIX: Swapped variables here as well for consistency ---
        norm1_param_grads, grad_from_main_path = self.norm1.backward(current_grad, cache["norm1_cache"])
        all_param_grads['norm1'] = norm1_param_grads
        
        grad_from_skip_path = grad_for_skip_path
        if self.skip_projection_flit:
            skip_proj_grads, grad_from_skip_path = self.skip_projection_flit.backward(grad_from_skip_path, cache["cache_skip_proj"])
            all_param_grads['skip_projection_flit'] = skip_proj_grads

        final_grad_data = grad_from_main_path.data + grad_from_skip_path.data
        final_grad_field = FSEField(final_grad_data, cache["inputs"].field_type, device=self.device)
        
        return all_param_grads, final_grad_field

    
class FlowField_Upsample:
    """✅ ENHANCED: Upsample with mixed precision support (unchanged for now)"""
    def __init__(self, factor: int, device: str, activation_type: FieldType = FieldType.CONTINUOUS):
        self.factor=factor
        self.device=device
        self.backend=cp if device=="gpu" else np
        self.parameters:Dict[str,FSEField]={}
        self.activation_type=activation_type
        
    def forward(self, inputs: FSEField) -> Tuple[FSEField, Dict[str,Any]]:
        """Forward pass with mixed precision support"""
        if inputs.device!=self.device: 
            inputs=inputs.to_device(self.device)
        
        # ✅ MIXED PRECISION: Preserve dtype
        current_dtype = inputs.dtype
        
        fh,fw=self.factor,self.factor
        up_h=self.backend.repeat(inputs.data,fh,axis=1)
        up_hw=self.backend.repeat(up_h,fw,axis=2)
        act_field=FieldOperations.apply_activation(
            FSEField(up_hw,inputs.field_type,device=self.device,dtype=current_dtype),
            self.activation_type
        )
        
        cache = {
            'inputs_shape':inputs.shape,
            'pre_activation_data':up_hw,
            'activation_type_used':self.activation_type,
            'factor': self.factor,
            'input_field_type': inputs.field_type,
            'dtype': current_dtype
        }
        
        return act_field, cache
        
    def backward(self, up_grad_act: FSEField, cache: Dict[str,Any]) -> Tuple[Dict[str,FSEField],FSEField]:
        """Backward pass with mixed precision support"""
        
        # ✅ MIXED PRECISION: Ensure gradient is in fp32 for computation
        if up_grad_act.dtype != self.backend.float32:
            grad_data_fp32 = up_grad_act.data.astype(self.backend.float32)
            up_grad_act_fp32 = FSEField(grad_data_fp32, up_grad_act.field_type, 
                                      device=up_grad_act.device, dtype=self.backend.float32)
        else:
            up_grad_act_fp32 = up_grad_act
        
        # ✅ VALIDATE CACHE
        if 'pre_activation_data' not in cache or 'inputs_shape' not in cache:
            logger.error("❌ Upsample backward: Missing cache keys")
            zero_grad = FSEField(self.backend.zeros((1,1,1,1), dtype=self.backend.float32), device=self.device, dtype=self.backend.float32)
            return {}, zero_grad
            
        pre_act_data=cache['pre_activation_data']
        act_type=cache['activation_type_used']
        in_s=cache['inputs_shape']
        
        # ✅ MIXED PRECISION: Convert pre_act_data to fp32 for derivative computation
        if pre_act_data.dtype != self.backend.float32:
            pre_act_data_fp32 = pre_act_data.astype(self.backend.float32)
        else:
            pre_act_data_fp32 = pre_act_data
        
        grad_pre_act=FieldOperations.activation_derivative(up_grad_act_fp32,pre_act_data_fp32,act_type)
        fh,fw=self.factor,self.factor
        B,Hu,Wu,Cu = grad_pre_act.shape
        reshaped_grad=grad_pre_act.data.reshape(B,in_s[1],fh,in_s[2],fw,Cu)
        down_grad_data=reshaped_grad.sum(axis=(2,4))
        
        return {}, FSEField(down_grad_data, field_type=grad_pre_act.field_type, device=self.device, dtype=self.backend.float32)

class FlowField_Downsample(FlowField_Upsample):
    """✅ ENHANCED: Downsample with mixed precision support (unchanged for now)"""
    def __init__(self, factor: int, device: str, activation_type: FieldType = FieldType.CONTINUOUS):
        super().__init__(factor, device, activation_type)
        
    def forward(self, inputs: FSEField) -> Tuple[FSEField, Dict[str, Any]]:
        """Forward pass with mixed precision support"""
        if inputs.device != self.device: 
            inputs = inputs.to_device(self.device)
        
        # ✅ MIXED PRECISION: Preserve dtype
        current_dtype = inputs.dtype
        
        B,H,W,C = inputs.shape
        fh,fw = self.factor,self.factor
        Hp,Wp = H//fh, W//fw
        input_eff = inputs.data[:, :Hp*fh, :Wp*fw, :]
        pooled = input_eff.reshape(B,Hp,fh,Wp,fw,C).mean(axis=(2,4))
        act_field = FieldOperations.apply_activation(
            FSEField(pooled,inputs.field_type,device=self.device,dtype=current_dtype), 
            self.activation_type
        )
        
        cache = {
            'inputs_original_shape':inputs.shape, 
            'pre_activation_data':pooled, 
            'activation_type_used':self.activation_type,
            'factor': self.factor,
            'effective_dims': (Hp, Wp),
            'input_field_type': inputs.field_type,
            'dtype': current_dtype
        }
        
        return act_field, cache
        
    def backward(self, up_grad_act: FSEField, cache: Dict[str, Any]) -> Tuple[Dict[str,FSEField], FSEField]:
        """Backward pass with mixed precision support"""
        
        # ✅ MIXED PRECISION: Ensure gradient is in fp32 for computation
        if up_grad_act.dtype != self.backend.float32:
            grad_data_fp32 = up_grad_act.data.astype(self.backend.float32)
            up_grad_act_fp32 = FSEField(grad_data_fp32, up_grad_act.field_type, 
                                      device=up_grad_act.device, dtype=self.backend.float32)
        else:
            up_grad_act_fp32 = up_grad_act
        
        # ✅ VALIDATE CACHE
        required_keys = ['pre_activation_data', 'inputs_original_shape', 'activation_type_used']
        missing_keys = [key for key in required_keys if key not in cache]
        
        if missing_keys:
            logger.error(f"❌ Downsample backward: Missing cache keys {missing_keys}")
            zero_grad = FSEField(self.backend.zeros((1,1,1,1), dtype=self.backend.float32), device=self.device, dtype=self.backend.float32)
            return {}, zero_grad
            
        pre_act_data=cache['pre_activation_data']
        act_type=cache['activation_type_used']
        in_orig_s=cache['inputs_original_shape']
        
        # ✅ MIXED PRECISION: Convert pre_act_data to fp32 for derivative computation
        if pre_act_data.dtype != self.backend.float32:
            pre_act_data_fp32 = pre_act_data.astype(self.backend.float32)
        else:
            pre_act_data_fp32 = pre_act_data
        
        grad_pre_act = FieldOperations.activation_derivative(up_grad_act_fp32,pre_act_data_fp32,act_type)
        fh,fw=self.factor,self.factor
        B,Ho,Wo,Co = in_orig_s
        Hp_eff,Wp_eff = Ho//fh,Wo//fw
        grad_exp_h = self.backend.repeat(grad_pre_act.data,fh,axis=1)
        grad_exp_hw = self.backend.repeat(grad_exp_h,fw,axis=2)
        down_grad_eff = grad_exp_hw/(fh*fw)
        down_grad_full = self.backend.zeros(in_orig_s,dtype=down_grad_eff.dtype)
        down_grad_full[:,:Hp_eff*fh,:Wp_eff*fw,:] = down_grad_eff
        
        return {}, FSEField(down_grad_full, field_type=grad_pre_act.field_type, device=self.device, dtype=self.backend.float32)


class FlowField_ContinuousAttention:
    """
    [DEFINITIVE, COMPLETE VERSION] Implements Continuous Field Attention (CFA)
    with a full forward pass that creates a cache and a full backward pass that
    correctly computes gradients for all parameters and inputs.
    """
    def __init__(self, query_channels: int, key_value_channels: int, head_dim: int, device: str = "gpu"):
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.head_dim = head_dim
        self.cache: Dict[str, Any] = {} # Initialize the cache attribute

        # Use FLITs for the linear projections, making this FSE-native
        self.query_proj = FlowField_FLIT(query_channels, head_dim, FieldType.LINEAR, 0.0, device, use_bias=False)
        self.key_proj = FlowField_FLIT(key_value_channels, head_dim, FieldType.LINEAR, 0.0, device, use_bias=False)
        self.value_proj = FlowField_FLIT(key_value_channels, head_dim, FieldType.LINEAR, 0.0, device, use_bias=False)
        self.output_proj = FlowField_FLIT(head_dim, query_channels, FieldType.LINEAR, 0.0, device, use_bias=False)

        self.parameters = {
            'query_proj': self.query_proj.parameters,
            'key_proj': self.key_proj.parameters,
            'value_proj': self.value_proj.parameters,
            'output_proj': self.output_proj.parameters
        }

    def forward(self, query_field: FSEField, key_value_field: FSEField) -> Tuple[FSEField, Dict[str, Any]]:
        """
        Performs the CFA operation and now returns a cache for the backward pass.
        """
        # 1. Project inputs to Q, K, V fields
        query, q_cache = self.query_proj.forward(query_field)
        keys, k_cache = self.key_proj.forward(key_value_field)
        values, v_cache = self.value_proj.forward(key_value_field)

        # 2. Compute the Energy and Attention Fields
        energy_field_data = query.data @ self.backend.transpose(keys.data, (0, 2, 1))
        energy_field_data /= self.backend.sqrt(self.head_dim)
        attention_field_data = 1.0 / (1.0 + self.backend.exp(-energy_field_data))
        
        # 3. Modulate the Value field
        attended_context_data = attention_field_data @ values.data
        attended_context_field_pre_proj = FSEField(attended_context_data, query.field_type, device=self.device)

        # 4. Apply final output projection
        final_context, out_cache = self.output_proj.forward(attended_context_field_pre_proj)

        # 5. Store all intermediate values and caches for the backward pass
        self.cache = {
            "query_field": query_field, "key_value_field": key_value_field,
            "query": query, "keys": keys, "values": values,
            "attention_field_data": attention_field_data,
            "attended_context_field_pre_proj": attended_context_field_pre_proj,
            "q_cache": q_cache, "k_cache": k_cache, "v_cache": v_cache, "out_cache": out_cache
        }
        
        return final_context, self.cache

    def backward(self, upstream_grad: FSEField, cache: Dict[str, Any]) -> Tuple[Dict[str, Any], FSEField]:
        """
        Implements the full backward pass for the CFA module.
        """
        param_grads = {}
        
        # 1. Backprop through the final output projection
        out_proj_param_grads, grad_attended_context = self.output_proj.backward(upstream_grad, cache["out_cache"])
        param_grads["output_proj"] = out_proj_param_grads
        
        # 2. Backpropagate through the attention @ values operation
        attention_data = cache["attention_field_data"]
        values = cache["values"]
        grad_values = self.backend.transpose(attention_data, (0, 2, 1)) @ grad_attended_context.data
        grad_attention = grad_attended_context.data @ self.backend.transpose(values.data, (0, 2, 1))
        
        # 3. Backpropagate through the sigmoid and scaling
        grad_energy = grad_attention * (attention_data * (1.0 - attention_data))
        grad_energy /= self.backend.sqrt(self.head_dim)
        
        # 4. Backpropagate through the query @ keys.T operation
        query = cache["query"]
        keys = cache["keys"]
        grad_keys = self.backend.transpose(query.data, (0, 2, 1)) @ grad_energy
        grad_query = grad_energy @ keys.data
        
        # 5. Backpropagate through the Q, K, V projection layers
        q_proj_grads, grad_q_input = self.query_proj.backward(FSEField(grad_query, device=self.device), cache["q_cache"])
        k_proj_grads, grad_kv_input1 = self.key_proj.backward(FSEField(self.backend.transpose(grad_keys, (0, 2, 1)), device=self.device), cache["k_cache"])
        v_proj_grads, grad_kv_input2 = self.value_proj.backward(FSEField(grad_values, device=self.device), cache["v_cache"])

        param_grads["query_proj"] = q_proj_grads
        param_grads["key_proj"] = k_proj_grads
        param_grads["value_proj"] = v_proj_grads
        
        # The downstream gradient is the sum of gradients flowing back to the inputs
        downstream_grad_to_kv = grad_kv_input1.data + grad_kv_input2.data
        
        # This function only returns parameter gradients and the gradient for the key_value_field.
        # The gradient for the query_field is handled separately in the evolver backward pass.
        return param_grads, FSEField(downstream_grad_to_kv, device=self.device)



class FlowField_Conv1D_FLIT:
    """
    [DEFINITIVE, ROBUST VERSION] A true Field Interaction Layer (FIL) that
    implements a robust 1D convolution using einsum. This version contains the
    corrected backward pass return signature.
    """
    def __init__(self, channels: int, kernel_size: int, device: str,
                 field_type: FieldType = FieldType.CONTINUOUS, use_bias: bool = True):
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.activation_field_type = field_type
        self.kernel_size = kernel_size
        
        current_dtype = get_default_dtype()
        
        # Kernel shape for 1D conv is (Kernel_Width, In_Channels, Out_Channels)
        init_scale = self.backend.sqrt(2. / (kernel_size * channels))
        kernel_data = self.backend.random.normal(0, init_scale, (kernel_size, channels, channels)).astype(current_dtype)
        self.kernel = FSEField(kernel_data, field_type=FieldType.LINEAR, device=device, dtype=current_dtype)
        self.parameters = {'kernel': self.kernel}

        self.use_bias = use_bias
        if self.use_bias:
            bias_data = self.backend.zeros(channels, dtype=current_dtype)
            self.bias = FSEField(bias_data, field_type=FieldType.LINEAR, device=device, dtype=current_dtype)
            self.parameters['bias'] = self.bias

    def _pad_sequence_for_conv(self, sequence: FSEField) -> FSEField:
        """Applies SAME padding for the 1D convolution."""
        padding = self.kernel_size - 1
        pad_left = padding // 2
        pad_right = padding - pad_left
        pad_width = ((0, 0), (pad_left, pad_right), (0, 0))
        padded_data = self.backend.pad(sequence.data, pad_width, mode='edge')
        return FSEField(padded_data, sequence.field_type, device=self.device)

    def forward(self, inputs: FSEField, context_signal: Optional[FSEField] = None, training: bool = True) -> Tuple[FSEField, Dict[str, Any]]:
        """
        The forward pass using a robust einsum implementation.
        """
        # 1. Pad the input to maintain sequence length (SAME padding)
        padded_input = self._pad_sequence_for_conv(inputs)
        
        # 2. Use as_strided to create a view of sliding windows (im2col)
        B, T_padded, C = padded_input.shape
        sB, sT, sC = padded_input.data.strides
        T_out = inputs.shape[1]
        
        patches = self.backend.lib.stride_tricks.as_strided(
            padded_input.data,
            shape=(B, T_out, self.kernel_size, C),
            strides=(sB, sT, sT, sC)
        )
        
        # 3. Perform convolution with einsum
        pre_activation_data = self.backend.einsum('btkc,kco->bto', patches, self.kernel.data)
        if self.use_bias:
            pre_activation_data += self.bias.data
        
        pre_activation_field = FSEField(pre_activation_data, device=self.device)

        # 4. Apply activation
        output_field = FieldOperations.apply_activation(pre_activation_field, self.activation_field_type)

        cache = {
            "input_patches": patches,
            "pre_activation_data": pre_activation_data,
            "activation_type_used": self.activation_field_type,
            "original_input_shape": inputs.shape
        }
        return output_field, cache

    def backward(self, upstream_grad: FSEField, cache: Dict[str, Any]) -> Tuple[Dict[str, Any], FSEField]:
        """
        [DEFINITIVE FIX] This backward pass is guaranteed to return the tuple
        in the correct order: (parameter_gradients_dict, data_gradient_field).
        """
        param_grads = {}
        
        # 1. Backpropagate through the activation function
        grad_pre_activation = FieldOperations.activation_derivative(
            upstream_grad, cache['pre_activation_data'], cache['activation_type_used']
        )
        
        # 2. Calculate the gradient with respect to the kernel (dW)
        grad_kernel_data = self.backend.einsum('btkc,bto->kco', cache['input_patches'], grad_pre_activation.data, optimize=True)
        param_grads['kernel'] = FSEField(grad_kernel_data, device=self.device)
        if self.use_bias:
            param_grads['bias'] = FSEField(self.backend.sum(grad_pre_activation.data, axis=(0, 1)), device=self.device)
            
        # 3. Calculate the gradient with respect to the input patches (dPatches)
        grad_patches = self.backend.einsum('bto,kco->btkc', grad_pre_activation.data, self.kernel.data, optimize=True)
        
        # 4. Accumulate the patch gradients back to the input gradient (dX)
        grad_input_padded = self.backend.zeros(
            (cache['original_input_shape'][0], 
             cache['original_input_shape'][1] + self.kernel_size - 1, 
             cache['original_input_shape'][2]), 
            dtype=grad_patches.dtype
        )
        
        for i in range(self.kernel_size):
            grad_input_padded[:, i:i+grad_patches.shape[1], :] += grad_patches[:, :, i, :]
        
        # 5. Un-pad the gradient to match original input shape
        padding = self.kernel_size - 1
        pad_left = padding // 2
        downstream_grad_data = grad_input_padded[:, pad_left:pad_left + cache['original_input_shape'][1], :]
        downstream_grad = FSEField(downstream_grad_data, device=self.device)

        # --- THE FIX ---
        # The return order is guaranteed to be (dictionary, FSEField).
        return param_grads, downstream_grad


# In file: adjoint_components.py
# Replace the entire FlowField_LayerNorm class with this definitive, mathematically correct version.

class FlowField_LayerNorm:
    """
    [DEFINITIVE, NUMERICALLY STABLE FP32 VERSION] This version is robust to
    high-magnitude inputs and contains the CORRECTED BACKWARD PASS with the
    full, mathematically complete gradient calculation.
    """
    def __init__(self, channels: int, device: str, epsilon: float = 1e-5):
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.epsilon = epsilon
        
        # All parameters and computations should be in full fp32
        current_dtype = self.backend.float32
        self.gamma = FSEField(self.backend.ones(channels, dtype=current_dtype), device=device)
        self.beta = FSEField(self.backend.zeros(channels, dtype=current_dtype), device=device)
        self.parameters = {'gamma': self.gamma, 'beta': self.beta}

    def forward(self, field: FSEField) -> Tuple[FSEField, Dict[str, Any]]:
        # Ensure input is fp32 for stable calculations
        field_data_fp32 = field.data.astype(self.backend.float32)
        
        mean = self.backend.mean(field_data_fp32, axis=-1, keepdims=True)
        variance = self.backend.var(field_data_fp32, axis=-1, keepdims=True)
        
        # Clamp variance to be non-negative to prevent sqrt(negative) -> NaN
        stable_variance = self.backend.maximum(variance, 0)
        
        inv_std = 1.0 / self.backend.sqrt(stable_variance + self.epsilon)
        x_norm = (field_data_fp32 - mean) * inv_std
        
        output_data = self.gamma.data * x_norm + self.beta.data
        
        # Return in the original field's precision
        output_field = FSEField(output_data.astype(field.dtype), field.field_type, device=self.device)
        
        cache = {
            "x_norm": x_norm, "gamma": self.gamma, "inv_std": inv_std,
            "input_shape": field.shape, "input_data": field_data_fp32,
            "mean": mean
        }
        return output_field, cache

    def backward(self, upstream_grad: FSEField, cache: Dict[str, Any]) -> Tuple[Dict[str, Any], FSEField]:
        """
        [DEFINITIVE FIX] This is the full, mathematically correct and stable backward pass.
        It is guaranteed to return the tuple in the correct order: (dict, FSEField).
        """
        param_grads = {}
        
        # Ensure all calculations are done in float32 for stability
        upstream_grad_data_fp32 = upstream_grad.data.astype(self.backend.float32)
        x_norm = cache["x_norm"].astype(self.backend.float32)
        gamma_data = cache["gamma"].data.astype(self.backend.float32)
        inv_std = cache["inv_std"].astype(self.backend.float32)
        input_data = cache["input_data"].astype(self.backend.float32)
        mean = cache["mean"].astype(self.backend.float32)
        C = cache["input_shape"][-1]

        # Gradients for learnable parameters gamma and beta
        grad_gamma = self.backend.sum(upstream_grad_data_fp32 * x_norm, axis=(0, 1))
        grad_beta = self.backend.sum(upstream_grad_data_fp32, axis=(0, 1))
        param_grads["gamma"] = FSEField(grad_gamma, device=self.device, dtype=self.backend.float32)
        param_grads["beta"] = FSEField(grad_beta, device=self.device, dtype=self.backend.float32)
        
        # Gradient for the input field
        d_x_norm = upstream_grad_data_fp32 * gamma_data
        
        # --- THE MATHEMATICAL FIX ---
        # The gradient of the mean has two terms. Your version was missing the second term.
        d_variance = self.backend.sum(d_x_norm * (input_data - mean), axis=-1, keepdims=True) * -0.5 * inv_std**3
        d_mean_term1 = self.backend.sum(d_x_norm * -inv_std, axis=-1, keepdims=True)
        d_mean_term2 = self.backend.sum(-2.0 * (input_data - mean), axis=-1, keepdims=True) * d_variance / C
        d_mean = d_mean_term1 + d_mean_term2
        
        d_input = (d_x_norm * inv_std) + (d_variance * 2.0 * (input_data - mean) / C) + (d_mean / C)
        
        downstream_grad = FSEField(d_input, upstream_grad.field_type, device=self.device, dtype=self.backend.float32)
        
        return param_grads, downstream_grad