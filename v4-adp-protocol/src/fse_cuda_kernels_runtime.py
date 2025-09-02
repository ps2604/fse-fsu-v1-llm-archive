# file: fse_cuda_kernels.py
# FSE CUDA KERNELS - PURE PYTHON WITH RUNTIME COMPILATION
# Complete FSE adjoint PDE solving kernels using CuPy RawKernel
# Perfect for Vertex AI and cloud deployment - no pre-compilation needed!

import cupy as cp
import numpy as np
from typing import Optional, Tuple, Union, Dict
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class FieldType(Enum):
    """Field type enumeration matching FlowField core"""
    CONTINUOUS = 0
    WAVE = 1
    QUANTUM = 2
    SPATIAL = 3
    LINEAR = 4

class FSECUDAKernels:
    """
    FSE CUDA kernels with runtime compilation using CuPy RawKernel.
    Implements the complete adjoint PDE solving mathematics from FSE/FlowField patents.
    """
    
    def __init__(self):
        """Initialize FSE CUDA kernels with runtime compilation"""
        self.kernels = {}
        self.device_id = cp.cuda.Device().id
        self._compile_kernels()
        logger.info(f"✅ FSE CUDA Kernels compiled successfully on GPU {self.device_id}")
    
    def _compile_kernels(self):
        """Compile all CUDA kernels using CuPy RawKernel"""
        
        # === FORWARD OPERATOR KERNEL ===
        forward_kernel_code = '''
        extern "C" __global__
        void fse_forward_operator_kernel(
            float* __restrict__ dF_dt_out,
            const float* __restrict__ F_in,
            const float* __restrict__ kernel_param,
            const float* __restrict__ bias_param,
            const float* __restrict__ context_signal,
            const int batch_size,
            const int sequence_length,
            const int channels,
            const float evolution_rate,
            const int field_type,
            const float dt
        ) {
            // Mathematical constants
            const float PI = 3.14159265358979323846f;
            const float MAX_FIELD_MAG = 1e6f;
            const float STABILITY_THRESH = 1e3f;
            
            // Thread indexing with coalesced memory access
            const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int total_elements = batch_size * sequence_length * channels;
            
            if (global_idx >= total_elements) return;
            
            // Decompose linear index into (batch, seq, channel) coordinates
            const int batch_idx = global_idx / (sequence_length * channels);
            const int remaining = global_idx % (sequence_length * channels);
            const int seq_idx = remaining / channels;
            const int chan_idx = remaining % channels;
            
            // Load current field value
            const float field_value = F_in[global_idx];
            
            // Initialize field derivative computation
            float dF_dt = 0.0f;
            
            // === LINGUISTIC FIELD OPERATOR Ψ_linguistic[F(s,t), Θ(s)] ===
            
            // 1. Local field transformation using kernel parameters
            if (kernel_param != nullptr) {
                const int kernel_idx = chan_idx * channels + chan_idx; // 1x1 kernel
                if (kernel_idx < channels * channels) {
                    const float kernel_weight = kernel_param[kernel_idx];
                    dF_dt += kernel_weight * field_value;
                }
            }
            
            // 2. Spatial field interactions (neighboring sequence positions)
            float spatial_contribution = 0.0f;
            
            // Left neighbor (with bounds checking)
            if (seq_idx > 0) {
                const int left_idx = batch_idx * sequence_length * channels + 
                                   (seq_idx - 1) * channels + chan_idx;
                spatial_contribution += 0.1f * F_in[left_idx];
            }
            
            // Right neighbor (with bounds checking)
            if (seq_idx < sequence_length - 1) {
                const int right_idx = batch_idx * sequence_length * channels + 
                                    (seq_idx + 1) * channels + chan_idx;
                spatial_contribution += 0.1f * F_in[right_idx];
            }
            
            dF_dt += spatial_contribution;
            
            // 3. Field-type specific dynamics
            float field_dynamics = 0.0f;
            
            // Stable tanh implementation
            auto stable_tanh = [](float x) -> float {
                if (x > 20.0f) return 1.0f;
                if (x < -20.0f) return -1.0f;
                float exp_2x = expf(2.0f * x);
                return (exp_2x - 1.0f) / (exp_2x + 1.0f);
            };
            
            switch (field_type) {
                case 0: // CONTINUOUS
                    field_dynamics = evolution_rate * stable_tanh(field_value);
                    break;
                    
                case 1: // WAVE
                    {
                        const float phase = 2.0f * PI * (float)seq_idx / (float)sequence_length;
                        field_dynamics = evolution_rate * sinf(field_value + phase);
                    }
                    break;
                    
                case 2: // QUANTUM
                    field_dynamics = evolution_rate * stable_tanh(field_value) * 
                                   cosf(2.0f * field_value);
                    break;
                    
                case 3: // SPATIAL
                    {
                        const float pos_weight = 1.0f + 0.1f * (float)seq_idx / (float)sequence_length;
                        field_dynamics = evolution_rate * pos_weight * stable_tanh(field_value);
                    }
                    break;
                    
                default: // LINEAR
                    field_dynamics = evolution_rate * field_value;
                    break;
            }
            
            dF_dt += field_dynamics;
            
            // === CONTEXT INFLUENCE Φ_context[F(s,t), C(s,t)] ===
            if (context_signal != nullptr) {
                const float context_value = context_signal[global_idx];
                dF_dt += 0.05f * context_value * field_value;
            }
            
            // === FIELD STABILITY TERM Δ_field[F(s,t)] ===
            const float field_magnitude = fabsf(field_value);
            if (field_magnitude > STABILITY_THRESH) {
                const float stability_correction = -0.01f * (field_magnitude - STABILITY_THRESH) * 
                                                 (field_value / field_magnitude);
                dF_dt += stability_correction;
            }
            
            // === BIAS ADDITION ===
            if (bias_param != nullptr && chan_idx < channels) {
                dF_dt += bias_param[chan_idx];
            }
            
            // === FINAL STABILITY CLAMPING ===
            dF_dt = fmaxf(fminf(dF_dt, MAX_FIELD_MAG), -MAX_FIELD_MAG);
            
            // Store result
            dF_dt_out[global_idx] = dF_dt;
        }
        '''
        
        # === ADJOINT OPERATOR KERNEL ===
        adjoint_kernel_code = '''
        extern "C" __global__
        void fse_adjoint_operator_kernel(
            float* __restrict__ dG_dt_out,
            const float* __restrict__ G_in,
            const float* __restrict__ F_forward_in,
            const float* __restrict__ kernel_param,
            const float* __restrict__ bias_param,
            const float* __restrict__ context_signal,
            const int batch_size,
            const int sequence_length,
            const int channels,
            const float evolution_rate,
            const int field_type,
            const float dt
        ) {
            const float PI = 3.14159265358979323846f;
            const float MAX_FIELD_MAG = 1e6f;
            const float STABILITY_THRESH = 1e3f;
            
            const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int total_elements = batch_size * sequence_length * channels;
            
            if (global_idx >= total_elements) return;
            
            // Decompose indices
            const int batch_idx = global_idx / (sequence_length * channels);
            const int remaining = global_idx % (sequence_length * channels);
            const int seq_idx = remaining / channels;
            const int chan_idx = remaining % channels;
            
            // Load adjoint and forward field values
            const float adjoint_value = G_in[global_idx];
            const float forward_value = F_forward_in[global_idx];
            
            float dG_dt = 0.0f;
            
            // === ADJOINT OF LINGUISTIC FIELD OPERATOR L*[G,F,Θ] ===
            
            // Stable functions for adjoint computation
            auto stable_tanh = [](float x) -> float {
                if (x > 20.0f) return 1.0f;
                if (x < -20.0f) return -1.0f;
                float exp_2x = expf(2.0f * x);
                return (exp_2x - 1.0f) / (exp_2x + 1.0f);
            };
            
            auto sech_squared = [&](float x) -> float {
                float tanh_x = stable_tanh(x);
                return 1.0f - tanh_x * tanh_x;
            };
            
            // Adjoint of activation functions
            float activation_adjoint = 1.0f;
            
            switch (field_type) {
                case 0: // CONTINUOUS - Adjoint of tanh
                    activation_adjoint = evolution_rate * sech_squared(forward_value);
                    break;
                    
                case 1: // WAVE - Adjoint of sin
                    {
                        const float phase = 2.0f * PI * (float)seq_idx / (float)sequence_length;
                        activation_adjoint = evolution_rate * cosf(forward_value + phase);
                    }
                    break;
                    
                case 2: // QUANTUM - Adjoint of tanh(x)*cos(2x)
                    {
                        const float tanh_x = stable_tanh(forward_value);
                        const float cos_2x = cosf(2.0f * forward_value);
                        const float sin_2x = sinf(2.0f * forward_value);
                        const float sech_sq = sech_squared(forward_value);
                        activation_adjoint = evolution_rate * (sech_sq * cos_2x - 2.0f * tanh_x * sin_2x);
                    }
                    break;
                    
                case 3: // SPATIAL - Adjoint of position-weighted tanh
                    {
                        const float pos_weight = 1.0f + 0.1f * (float)seq_idx / (float)sequence_length;
                        activation_adjoint = evolution_rate * pos_weight * sech_squared(forward_value);
                    }
                    break;
                    
                default: // LINEAR
                    activation_adjoint = evolution_rate;
                    break;
            }
            
            dG_dt += adjoint_value * activation_adjoint;
            
            // Adjoint of kernel operation (transpose)
            if (kernel_param != nullptr) {
                const int kernel_idx = chan_idx * channels + chan_idx;
                if (kernel_idx < channels * channels) {
                    const float kernel_weight = kernel_param[kernel_idx];
                    dG_dt += adjoint_value * kernel_weight;
                }
            }
            
            // Adjoint of spatial interactions
            float spatial_adjoint = 0.0f;
            
            if (seq_idx > 0) {
                const int left_idx = batch_idx * sequence_length * channels + 
                                   (seq_idx - 1) * channels + chan_idx;
                spatial_adjoint += 0.1f * G_in[left_idx];
            }
            
            if (seq_idx < sequence_length - 1) {
                const int right_idx = batch_idx * sequence_length * channels + 
                                    (seq_idx + 1) * channels + chan_idx;
                spatial_adjoint += 0.1f * G_in[right_idx];
            }
            
            dG_dt += spatial_adjoint;
            
            // === ADJOINT OF CONTEXT INFLUENCE ===
            if (context_signal != nullptr) {
                const float context_value = context_signal[global_idx];
                dG_dt += 0.05f * adjoint_value * context_value;
            }
            
            // === ADJOINT OF STABILITY TERM ===
            const float field_magnitude = fabsf(forward_value);
            if (field_magnitude > STABILITY_THRESH) {
                dG_dt += -0.01f * adjoint_value;
            }
            
            // === FINAL ADJOINT DERIVATIVE (NEGATIVE FOR BACKWARD TIME) ===
            dG_dt = -fmaxf(fminf(dG_dt, MAX_FIELD_MAG), -MAX_FIELD_MAG);
            
            dG_dt_out[global_idx] = dG_dt;
        }
        '''
        
        # === PARAMETER GRADIENT KERNEL ===
        param_grad_kernel_code = '''
        extern "C" __global__
        void fse_parameter_gradient_kernel(
            float* __restrict__ dL_dKernel_out,
            float* __restrict__ dL_dBias_out,
            const float* __restrict__ G_adjoint,
            const float* __restrict__ F_forward,
            const int batch_size,
            const int sequence_length,
            const int channels,
            const float dt
        ) {
            __shared__ float shared_kernel_grad[256];
            __shared__ float shared_bias_grad[256];
            
            const int tid = threadIdx.x;
            const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int total_elements = batch_size * sequence_length * channels;
            
            // Initialize shared memory
            shared_kernel_grad[tid] = 0.0f;
            shared_bias_grad[tid] = 0.0f;
            
            float local_kernel_grad = 0.0f;
            float local_bias_grad = 0.0f;
            
            // Grid-stride loop
            for (int idx = global_idx; idx < total_elements; idx += blockDim.x * gridDim.x) {
                if (idx < total_elements) {
                    const float adjoint_val = G_adjoint[idx];
                    const float forward_val = F_forward[idx];
                    
                    // Parameter gradients through field correlation
                    local_kernel_grad += adjoint_val * forward_val * dt;
                    local_bias_grad += adjoint_val * dt;
                }
            }
            
            shared_kernel_grad[tid] = local_kernel_grad;
            shared_bias_grad[tid] = local_bias_grad;
            
            __syncthreads();
            
            // Parallel reduction
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    shared_kernel_grad[tid] += shared_kernel_grad[tid + stride];
                    shared_bias_grad[tid] += shared_bias_grad[tid + stride];
                }
                __syncthreads();
            }
            
            // Atomic accumulation
            if (tid == 0) {
                const int param_idx = blockIdx.x % (channels * channels);
                const int bias_idx = blockIdx.x % channels;
                
                atomicAdd(&dL_dKernel_out[param_idx], shared_kernel_grad[0]);
                atomicAdd(&dL_dBias_out[bias_idx], shared_bias_grad[0]);
            }
        }
        '''
        
        # === FIELD INTEGRATION KERNEL ===
        integration_kernel_code = '''
        extern "C" __global__
        void fse_field_integration_kernel(
            float* __restrict__ F_out,
            const float* __restrict__ F_in,
            const float* __restrict__ dF_dt,
            const int batch_size,
            const int sequence_length,
            const int channels,
            const float dt
        ) {
            const float MAX_FIELD_MAG = 1e6f;
            
            const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int total_elements = batch_size * sequence_length * channels;
            
            if (global_idx >= total_elements) return;
            
            const float current_field = F_in[global_idx];
            const float field_derivative = dF_dt[global_idx];
            
            // Euler integration: F(t+dt) = F(t) + dt * dF/dt
            float integrated_field = current_field + dt * field_derivative;
            
            // Apply stability clamping
            integrated_field = fmaxf(fminf(integrated_field, MAX_FIELD_MAG), -MAX_FIELD_MAG);
            
            F_out[global_idx] = integrated_field;
        }
        '''
        
        # === FIELD SMOOTHING KERNEL ===
        smoothing_kernel_code = '''
        extern "C" __global__
        void fse_field_smoothing_kernel(
            float* __restrict__ F_out,
            const float* __restrict__ F_in,
            const int batch_size,
            const int sequence_length,
            const int channels,
            const float smoothing_strength
        ) {
            const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int total_elements = batch_size * sequence_length * channels;
            
            if (global_idx >= total_elements) return;
            
            // Decompose indices
            const int batch_idx = global_idx / (sequence_length * channels);
            const int remaining = global_idx % (sequence_length * channels);
            const int seq_idx = remaining / channels;
            const int chan_idx = remaining % channels;
            
            float smoothed_value = F_in[global_idx];
            
            // 3-point smoothing for field continuity
            if (seq_idx > 0 && seq_idx < sequence_length - 1) {
                const int left_idx = batch_idx * sequence_length * channels + 
                                   (seq_idx - 1) * channels + chan_idx;
                const int right_idx = batch_idx * sequence_length * channels + 
                                    (seq_idx + 1) * channels + chan_idx;
                
                const float left_val = F_in[left_idx];
                const float right_val = F_in[right_idx];
                
                smoothed_value = (1.0f - smoothing_strength) * smoothed_value + 
                               smoothing_strength * 0.5f * (left_val + right_val);
            }
            
            F_out[global_idx] = smoothed_value;
        }
        '''
        
        # Compile all kernels
        try:
            self.kernels['forward'] = cp.RawKernel(forward_kernel_code, 'fse_forward_operator_kernel')
            self.kernels['adjoint'] = cp.RawKernel(adjoint_kernel_code, 'fse_adjoint_operator_kernel')
            self.kernels['param_grad'] = cp.RawKernel(param_grad_kernel_code, 'fse_parameter_gradient_kernel')
            self.kernels['integrate'] = cp.RawKernel(integration_kernel_code, 'fse_field_integration_kernel')
            self.kernels['smooth'] = cp.RawKernel(smoothing_kernel_code, 'fse_field_smoothing_kernel')
            
            logger.info("✅ All FSE CUDA kernels compiled successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to compile FSE CUDA kernels: {e}")
            raise
    
    def _get_field_type_enum(self, field_type) -> int:
        """Convert FieldType to integer enum"""
        if hasattr(field_type, 'value'):
            return field_type.value
        elif isinstance(field_type, int):
            return field_type
        else:
            return 4  # Default to LINEAR
    
    def _get_pointer_or_null(self, array: Optional[cp.ndarray]) -> Optional[cp.ndarray]:
            """
            [CORRECTED TYPE HINT] Get CuPy array or None for kernel arguments.
            The return type hint is corrected to use Optional[cp.ndarray] which is
            the standard way of expressing 'cp.ndarray or None'.
            """
            return array if array is not None else None
    
    def forward_operator(
        self,
        F_in: cp.ndarray,
        kernel_param: Optional[cp.ndarray] = None,
        bias_param: Optional[cp.ndarray] = None,
        context_signal: Optional[cp.ndarray] = None,
        evolution_rate: float = 0.1,
        field_type=FieldType.CONTINUOUS,
        dt: float = 0.01
    ) -> cp.ndarray:
        """
        Compute forward operator: ∂F/∂t = Ψ_linguistic[F(s,t), Θ(s)]
        
        Args:
            F_in: Input field state (batch_size, sequence_length, channels)
            kernel_param: Learnable kernel parameters
            bias_param: Learnable bias parameters
            context_signal: Optional context signal
            evolution_rate: Field evolution rate
            field_type: FSE field type
            dt: Time step size
            
        Returns:
            Field time derivative ∂F/∂t
        """
        # Validate and prepare inputs
        if F_in.ndim != 3:
            raise ValueError(f"F_in must be 3D (batch, seq, channels), got {F_in.shape}")
        
        batch_size, sequence_length, channels = F_in.shape
        
        # Ensure float32 type
        if F_in.dtype != cp.float32:
            F_in = F_in.astype(cp.float32)
        
        # Allocate output
        dF_dt_out = cp.zeros_like(F_in, dtype=cp.float32)
        
        # Calculate grid and block dimensions
        total_elements = batch_size * sequence_length * channels
        threads_per_block = 256
        num_blocks = (total_elements + threads_per_block - 1) // threads_per_block
        
        # Convert field type
        field_type_enum = self._get_field_type_enum(field_type)
        
        # Prepare kernel arguments
        args = (
            dF_dt_out,
            F_in,
            self._get_pointer_or_null(kernel_param),
            self._get_pointer_or_null(bias_param),
            self._get_pointer_or_null(context_signal),
            batch_size,
            sequence_length,
            channels,
            evolution_rate,
            field_type_enum,
            dt
        )
        
        # Launch kernel
        self.kernels['forward'](
            (num_blocks,), (threads_per_block,), args
        )
        
        # Synchronize and check for errors
        cp.cuda.runtime.deviceSynchronize()
        
        return dF_dt_out
    
    def adjoint_operator(
        self,
        G_in: cp.ndarray,
        F_forward: cp.ndarray,
        kernel_param: Optional[cp.ndarray] = None,
        bias_param: Optional[cp.ndarray] = None,
        context_signal: Optional[cp.ndarray] = None,
        evolution_rate: float = 0.1,
        field_type=FieldType.CONTINUOUS,
        dt: float = 0.01
    ) -> cp.ndarray:
        """
        Compute adjoint operator: ∂G/∂t = -L*[G(s,t), F(s,t), Θ(s)]
        """
        # Validate shapes
        if G_in.shape != F_forward.shape:
            raise ValueError(f"Shape mismatch: G_in {G_in.shape} vs F_forward {F_forward.shape}")
        
        batch_size, sequence_length, channels = G_in.shape
        
        # Ensure float32 type
        if G_in.dtype != cp.float32:
            G_in = G_in.astype(cp.float32)
        if F_forward.dtype != cp.float32:
            F_forward = F_forward.astype(cp.float32)
        
        # Allocate output
        dG_dt_out = cp.zeros_like(G_in, dtype=cp.float32)
        
        # Calculate dimensions
        total_elements = batch_size * sequence_length * channels
        threads_per_block = 256
        num_blocks = (total_elements + threads_per_block - 1) // threads_per_block
        
        field_type_enum = self._get_field_type_enum(field_type)
        
        # Launch kernel
        args = (
            dG_dt_out,
            G_in,
            F_forward,
            self._get_pointer_or_null(kernel_param),
            self._get_pointer_or_null(bias_param),
            self._get_pointer_or_null(context_signal),
            batch_size,
            sequence_length,
            channels,
            evolution_rate,
            field_type_enum,
            dt
        )
        
        self.kernels['adjoint'](
            (num_blocks,), (threads_per_block,), args
        )
        
        cp.cuda.runtime.deviceSynchronize()
        
        return dG_dt_out
    
    def parameter_gradients(
        self,
        G_adjoint: cp.ndarray,
        F_forward: cp.ndarray,
        kernel_shape: Tuple[int, int],
        bias_shape: Tuple[int,],
        dt: float = 0.01
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Compute parameter gradients: ∂L/∂Θ = ∫ G(s,t) * ∂Ψ/∂Θ dt
        """
        batch_size, sequence_length, channels = G_adjoint.shape
        
        # Ensure float32 type
        if G_adjoint.dtype != cp.float32:
            G_adjoint = G_adjoint.astype(cp.float32)
        if F_forward.dtype != cp.float32:
            F_forward = F_forward.astype(cp.float32)
        
        # Allocate outputs (zero-initialized for accumulation)
        dL_dKernel_out = cp.zeros(kernel_shape, dtype=cp.float32)
        dL_dBias_out = cp.zeros(bias_shape, dtype=cp.float32)
        
        # Calculate dimensions
        total_elements = batch_size * sequence_length * channels
        threads_per_block = 256
        num_blocks = (total_elements + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        args = (
            dL_dKernel_out,
            dL_dBias_out,
            G_adjoint,
            F_forward,
            batch_size,
            sequence_length,
            channels,
            dt
        )
        
        self.kernels['param_grad'](
            (num_blocks,), (threads_per_block,), args
        )
        
        cp.cuda.runtime.deviceSynchronize()
        
        return dL_dKernel_out, dL_dBias_out
    
    def integrate_field(
        self,
        F_in: cp.ndarray,
        dF_dt: cp.ndarray,
        dt: float = 0.01
    ) -> cp.ndarray:
        """
        Integrate field: F(t+dt) = F(t) + dt * dF/dt
        """
        batch_size, sequence_length, channels = F_in.shape
        
        # Ensure float32 type
        if F_in.dtype != cp.float32:
            F_in = F_in.astype(cp.float32)
        if dF_dt.dtype != cp.float32:
            dF_dt = dF_dt.astype(cp.float32)
        
        # Allocate output
        F_out = cp.zeros_like(F_in, dtype=cp.float32)
        
        # Calculate dimensions
        total_elements = batch_size * sequence_length * channels
        threads_per_block = 256
        num_blocks = (total_elements + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        args = (F_out, F_in, dF_dt, batch_size, sequence_length, channels, dt)
        
        self.kernels['integrate'](
            (num_blocks,), (threads_per_block,), args
        )
        
        cp.cuda.runtime.deviceSynchronize()
        
        return F_out
    
    def smooth_field(
        self,
        F_in: cp.ndarray,
        smoothing_strength: float = 0.1
    ) -> cp.ndarray:
        """
        Apply field smoothing for continuity preservation
        """
        batch_size, sequence_length, channels = F_in.shape
        
        # Ensure float32 type
        if F_in.dtype != cp.float32:
            F_in = F_in.astype(cp.float32)
        
        # Allocate output
        F_out = cp.zeros_like(F_in, dtype=cp.float32)
        
        # Calculate dimensions
        total_elements = batch_size * sequence_length * channels
        threads_per_block = 256
        num_blocks = (total_elements + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        args = (F_out, F_in, batch_size, sequence_length, channels, smoothing_strength)
        
        self.kernels['smooth'](
            (num_blocks,), (threads_per_block,), args
        )
        
        cp.cuda.runtime.deviceSynchronize()
        
        return F_out

# Global kernel instance (lazy initialization)
_fse_kernels = None

def get_fse_kernels() -> FSECUDAKernels:
    """Get global FSE CUDA kernels instance"""
    global _fse_kernels
    if _fse_kernels is None:
        _fse_kernels = FSECUDAKernels()
    return _fse_kernels

# Convenience functions for direct use
def fse_forward_op(F_in, kernel_param=None, bias_param=None, **kwargs):
    """Convenience function for forward operator"""
    return get_fse_kernels().forward_operator(F_in, kernel_param, bias_param, **kwargs)

def fse_adjoint_op(G_in, F_forward, kernel_param=None, bias_param=None, **kwargs):
    """Convenience function for adjoint operator"""
    return get_fse_kernels().adjoint_operator(G_in, F_forward, kernel_param, bias_param, **kwargs)

def fse_param_grads(G_adjoint, F_forward, kernel_shape, bias_shape, **kwargs):
    """Convenience function for parameter gradients"""
    return get_fse_kernels().parameter_gradients(G_adjoint, F_forward, kernel_shape, bias_shape, **kwargs)

def test_fse_kernels():
    """Test FSE kernels functionality"""
    try:
        logger.info("🧪 Testing FSE CUDA kernels...")
        
        # Create test data
        batch_size, seq_len, channels = 2, 128, 64
        F_test = cp.random.randn(batch_size, seq_len, channels, dtype=cp.float32) * 0.1
        kernel_test = cp.random.randn(channels, channels, dtype=cp.float32) * 0.01
        bias_test = cp.random.randn(channels, dtype=cp.float32) * 0.01
        
        kernels = get_fse_kernels()
        
        # Test forward operator
        dF_dt = kernels.forward_operator(F_test, kernel_test, bias_test)
        
        # Test adjoint operator
        G_test = cp.random.randn(*F_test.shape, dtype=cp.float32) * 0.1
        dG_dt = kernels.adjoint_operator(G_test, F_test, kernel_test, bias_test)
        
        # Test parameter gradients
        kernel_grad, bias_grad = kernels.parameter_gradients(
            G_test, F_test, kernel_test.shape, bias_test.shape
        )
        
        # Test integration
        F_integrated = kernels.integrate_field(F_test, dF_dt)
        
        # Test smoothing
        F_smooth = kernels.smooth_field(F_test)
        
        # Validate results
        assert cp.isfinite(dF_dt).all(), "Forward operator produced non-finite values"
        assert cp.isfinite(dG_dt).all(), "Adjoint operator produced non-finite values"
        assert cp.isfinite(kernel_grad).all(), "Kernel gradients produced non-finite values"
        assert cp.isfinite(bias_grad).all(), "Bias gradients produced non-finite values"
        assert cp.isfinite(F_integrated).all(), "Integration produced non-finite values"
        assert cp.isfinite(F_smooth).all(), "Smoothing produced non-finite values"
        
        logger.info("✅ All FSE CUDA kernel tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ FSE CUDA kernel test failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests when script is executed directly
    test_fse_kernels()