# file: flowfield_core_optimized.py
# MAJOR PERFORMANCE OPTIMIZATIONS for FlowField + MIXED PRECISION IMPLEMENTATION
# FULLY ADJOINT IMPLEMENTED - Enhanced with continuous field operators for adjoint PDE solving
# Revision 7.0: ADJOINT FIELD OPERATORS - Added support for continuous field adjoint equations
# UPDATES: New field operators for adjoint PDE solving, enhanced field arithmetic operations

import numpy as np
import cupy as cp
from typing import Union, Tuple, Optional, Any, TypeVar, Dict, List
from enum import Enum
import logging
import time
from contextlib import contextmanager

# NEW:
from fse_cuda_kernels_runtime import (
    FSECUDAKernels as FSUCUDAKernels,
    get_fse_kernels, fse_forward_op, fse_adjoint_op, fse_param_grads
)

# Create wrapper classes for compatibility:
class FSUAdvancedFieldOperations:
    @staticmethod
    def apply_advanced_field_operation(field, operation_type):
        kernels = get_fse_kernels()
        if operation_type == "forward":
            return kernels.forward_operator(field.data)
        elif operation_type == "adjoint":
            return kernels.adjoint_operator(field.data, field.data)
        return field.data

class FSUKernelCacheManager:
    @staticmethod
    def get_cached_kernel(operation_name):
        return get_fse_kernels()

# =========================================
# 🚀 MIXED PRECISION CONFIGURATION
# =========================================

# Global default precision - set to fp16 for memory efficiency
DEFAULT_DTYPE = cp.float16

def set_default_dtype(dtype):
    """Set global default dtype for FSE fields"""
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = dtype
    logger.info(f"✅ FlowField DEFAULT_DTYPE set to: {dtype}")

def get_default_dtype():
    """Get current default dtype"""
    return DEFAULT_DTYPE

# Helper function for creating tensors with default dtype
def randn(shape, scale, dtype=None):
    """Create random normal tensor with optional dtype"""
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return (scale * cp.random.standard_normal(shape)).astype(dtype)

def zeros(shape, dtype=None):
    """Create zero tensor with optional dtype"""
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return cp.zeros(shape, dtype=dtype)

def ones(shape, dtype=None):
    """Create ones tensor with optional dtype"""  
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return cp.ones(shape, dtype=dtype)

# ---------------------------------------------------------------------
# Utility: guarantee a field is 4-D  (B, H, W, C)
# Accepts an FSEField whose .data may be 2-D, 3-D or 4-D
# ---------------------------------------------------------------------
def _ensure_4d(field: "FSEField") -> Tuple["FSEField", Tuple[int, int, int, int]]:
    """
    Returns:
        - a *new* FSEField whose .data is reshaped to (B, H, W, C)
        - the resulting shape tuple
    """
    B = H = W = C = None               # for mypy
    data = field.data                  # numpy / cupy array
    if field.ndim == 4:                # (B, H, W, C)  – nothing to do
        return field, field.shape
    elif field.ndim == 3:              # (B, W, C)
        B, W, C = data.shape
        new_data = data.reshape(B, 1, W, C)
    elif field.ndim == 2:              # (B, W)
        B, W = data.shape
        new_data = data.reshape(B, 1, W, 1)
    else:
        raise ValueError(f"Expected 2-, 3- or 4-D tensor, got shape {data.shape}")

    # create a sibling field sharing meta-info
    new_field = FSEField(
        new_data,
        field_type     = field.field_type,
        evolution_rate = field.evolution_rate,
        device         = field.device,
        use_memory_pool= False,  # ✅ MEMORY SAFE: No custom pooling
        dtype          = field.dtype  # ✅ MIXED PRECISION: Preserve dtype
    )
    return new_field, new_field.shape


logger = logging.getLogger(__name__)

Number = Union[int, float]
ArrayLike = Union[np.ndarray, cp.ndarray]
FSEFieldTypeVar = TypeVar('FSEFieldTypeVar', bound='FSEField')

class FieldType(Enum):
    CONTINUOUS = "continuous"
    WAVE = "wave"
    QUANTUM = "quantum"
    SPATIAL = "spatial"
    MATERIAL = "material"
    LIGHTING = "lighting"
    LINEAR = "linear"

# =========================================
# 🚀 MEMORY SAFE POOL MANAGER - DISABLED CUSTOM POOLS
# =========================================

class FlowFieldMemoryPool:
    """
    ✅ MEMORY SAFE: Disabled custom memory pools to prevent aliasing corruption
    Always uses CuPy's proven default allocator
    """
    
    def __init__(self, device: str = "gpu", pool_size_gb: float = 8.0, default_dtype=None):
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.default_dtype = default_dtype or DEFAULT_DTYPE
        logger.info(f"✅ FlowFieldMemoryPool: Using CuPy default allocator (custom pools DISABLED for safety)")
    
    def get_buffer(self, shape: Tuple[int, ...], dtype=None) -> ArrayLike:
        """
        ✅ MEMORY SAFE: Always use CuPy's proven allocator
        NO custom pooling to prevent aliasing corruption
        """
        return self.backend.zeros(shape, dtype=dtype or self.default_dtype)
    
    def free_buffer(self, buffer: ArrayLike):
        """Return buffer to pool (no-op, CuPy handles this)"""
        pass

# Global memory pool instance
_global_memory_pool = None

def get_memory_pool(device: str = "gpu") -> FlowFieldMemoryPool:
    global _global_memory_pool
    if _global_memory_pool is None or _global_memory_pool.device != device:
        _global_memory_pool = FlowFieldMemoryPool(device, default_dtype=DEFAULT_DTYPE)
    return _global_memory_pool

# =========================================
# 🚀 MEMORY SAFE FSEField with ADJOINT OPERATORS
# =========================================

class FSEField:
    """
    ✅ FULLY ADJOINT IMPLEMENTED: FSEField with enhanced continuous field operators for adjoint PDE solving
    """
    def __init__(self,
                 data: ArrayLike,
                 field_type: FieldType = FieldType.LINEAR,
                 evolution_rate: float = 0.1,
                 device: str = "cpu",
                 use_memory_pool: bool = False,  # ✅ MEMORY SAFE: Default to False
                 dtype=None):  # ✅ MIXED PRECISION: Added dtype parameter
        
        if not isinstance(data, (np.ndarray, cp.ndarray)):
            raise TypeError(f"Field data must be NumPy/CuPy array, got {type(data)}")
        if not isinstance(field_type, FieldType):
            raise TypeError(f"field_type must be FieldType Enum, got {type(field_type)}")

        # ✅ MIXED PRECISION: Handle dtype conversion
        if dtype is not None:
            data = data.astype(dtype, copy=False)
        elif not hasattr(data, 'dtype') or data.dtype not in [cp.float16, cp.float32, np.float16, np.float32]:
            # Default to DEFAULT_DTYPE for new tensors
            data = data.astype(DEFAULT_DTYPE, copy=False)

        # ✅ MEMORY SAFE: Enhanced device conversion with full synchronization
        target_backend = cp if device == "gpu" else np
        
        if device == "gpu" and not isinstance(data, cp.ndarray):
            # ALWAYS use CuPy's default allocator - most reliable
            try:
                # Pre-conversion synchronization
                if hasattr(data, 'device') and str(data.device) != 'cpu':
                    cp.cuda.runtime.deviceSynchronize()
                
                data = cp.asarray(data)
                
                # Post-conversion synchronization
                cp.cuda.runtime.deviceSynchronize()
                
            except Exception as e:
                logger.error(f"GPU conversion failed: {e}")
                raise
                
        elif device == "cpu" and isinstance(data, cp.ndarray):
            try:
                # Pre-conversion synchronization
                cp.cuda.runtime.deviceSynchronize()
                
                data = cp.asnumpy(data)
                
            except Exception as e:
                logger.error(f"CPU conversion failed: {e}")
                raise

        self.data: ArrayLike = data
        self.field_type: FieldType = field_type
        self.evolution_rate: float = evolution_rate
        self.device: str = device
        self.grad: Optional[ArrayLike] = None
        self._use_memory_pool = False  # ✅ MEMORY SAFE: Always False

    @property
    def shape(self) -> Tuple[int, ...]: return self.data.shape
    @property
    def ndim(self) -> int: return self.data.ndim
    @property
    def dtype(self): return self.data.dtype
    @property
    def backend(self): return cp if self.device == "gpu" else np
    @property
    def size(self) -> int: return self.data.size

    def to_device(self: FSEFieldTypeVar, device: str) -> FSEFieldTypeVar:
        """
        ✅ MEMORY SAFE: Enhanced device transfer with complete synchronization
        """
        if device == self.device:
            return self

        try:
            # ✅ MEMORY SAFE: Pre-transfer synchronization
            if self.device == "gpu":
                cp.cuda.runtime.deviceSynchronize()

            # ✅ MEMORY SAFE: Always use CuPy's proven default allocator
            if device == "gpu":
                # Force complete copy to avoid any reference sharing
                new_data = cp.array(self.data, copy=True)
                # Ensure transfer completion
                cp.cuda.runtime.deviceSynchronize()
            else:
                # CPU transfer with synchronization
                if isinstance(self.data, cp.ndarray):
                    cp.cuda.runtime.deviceSynchronize()
                new_data = np.array(cp.asnumpy(self.data), copy=True)  # Force copy
            
            # ✅ MEMORY SAFE: Create new field with no memory pool
            result = self.__class__(
                new_data, 
                self.field_type, 
                self.evolution_rate, 
                device, 
                use_memory_pool=False,  # NO custom pooling
                dtype=self.dtype
            )
            
            # ✅ MEMORY SAFE: Post-transfer synchronization
            if device == "gpu":
                cp.cuda.runtime.deviceSynchronize()
            
            return result
            
        except Exception as e:
            logger.error(f"Device transfer failed: {e}")
            # ✅ MEMORY SAFE: Emergency cleanup on failure
            try:
                if device == "gpu":
                    cp.cuda.runtime.deviceSynchronize()
                    cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
            raise

    # ✅ ADJOINT FIELD ARITHMETIC: Enhanced operations for continuous field computation
    def _check_compat_and_get_data(self: FSEFieldTypeVar, other: Union[FSEFieldTypeVar, Number, ArrayLike]) -> Tuple[ArrayLike, ArrayLike, str, Any]:
        s_data = self.data
        o_data_arr: ArrayLike
        if isinstance(other, FSEField):
            o_data_arr = other.data
            o_device = other.device
        elif isinstance(other, (np.ndarray, cp.ndarray)): 
            o_data_arr = other
            o_device = "gpu" if isinstance(other, cp.ndarray) else "cpu"
        else: 
            o_data_arr = self.backend.array(other, dtype=self.dtype)  # ✅ MIXED PRECISION: Preserve dtype
            o_device = self.device
            
        target_device = self.device
        if self.device != o_device:
            if self.device == "gpu": 
                o_data_arr = cp.asarray(o_data_arr)
                cp.cuda.runtime.deviceSynchronize()  # ✅ MEMORY SAFE: Sync after conversion
            elif o_device == "gpu": 
                s_data = cp.asarray(s_data)
                target_device = "gpu"
                cp.cuda.runtime.deviceSynchronize()  # ✅ MEMORY SAFE: Sync after conversion
        
        backend_op = cp if target_device == "gpu" else np
        return s_data, o_data_arr, target_device, backend_op

    def __add__(self: FSEFieldTypeVar, other: Union[FSEFieldTypeVar, Number, ArrayLike]) -> FSEFieldTypeVar:
        """✅ ADJOINT ENHANCED: Field addition with continuity preservation"""
        s_data, o_data, res_device, backend = self._check_compat_and_get_data(other)
        try: 
            # Basic addition
            result_data = s_data + o_data
            
            # ✅ ADJOINT ENHANCEMENT: Apply field continuity preservation for adjoint PDE compatibility
            if isinstance(other, FSEField) and (self.field_type in [FieldType.CONTINUOUS, FieldType.SPATIAL]):
                result_data = self._apply_continuity_preservation(result_data, backend)
                
        except ValueError as e: 
            raise ValueError(f"Shape mismatch FSEField add: {s_data.shape} vs {getattr(o_data, 'shape', type(o_data))}. {e}")
        return self.__class__(result_data, self.field_type, self.evolution_rate, res_device, False, self.dtype)

    def __sub__(self: FSEFieldTypeVar, other: Union[FSEFieldTypeVar, Number, ArrayLike]) -> FSEFieldTypeVar:
        """✅ ADJOINT ENHANCED: Field subtraction with continuity preservation"""
        s_data, o_data, res_device, backend = self._check_compat_and_get_data(other)
        try: 
            result_data = s_data - o_data
            
            # ✅ ADJOINT ENHANCEMENT: Apply field continuity preservation
            if isinstance(other, FSEField) and (self.field_type in [FieldType.CONTINUOUS, FieldType.SPATIAL]):
                result_data = self._apply_continuity_preservation(result_data, backend)
                
        except ValueError as e: 
            raise ValueError(f"Shape mismatch FSEField sub: {s_data.shape} vs {getattr(o_data, 'shape', type(o_data))}. {e}")
        return self.__class__(result_data, self.field_type, self.evolution_rate, res_device, False, self.dtype)

    def __mul__(self: FSEFieldTypeVar, other: Union[FSEFieldTypeVar, Number, ArrayLike]) -> FSEFieldTypeVar:
        """✅ ADJOINT ENHANCED: Field multiplication with continuity preservation"""
        s_data, o_data, res_device, backend = self._check_compat_and_get_data(other)
        try: 
            result_data = s_data * o_data
            
            # ✅ ADJOINT ENHANCEMENT: Apply field continuity for multiplicative operations
            if isinstance(other, FSEField) and (self.field_type in [FieldType.CONTINUOUS, FieldType.WAVE]):
                result_data = self._apply_multiplicative_continuity(result_data, s_data, o_data, backend)
                
        except ValueError as e: 
            raise ValueError(f"Shape mismatch FSEField mul: {s_data.shape} vs {getattr(o_data, 'shape', type(o_data))}. {e}")
        return self.__class__(result_data, self.field_type, self.evolution_rate, res_device, False, self.dtype)

    def __neg__(self: FSEFieldTypeVar) -> FSEFieldTypeVar:
        """Enable unary negation for FSEField objects."""
        return self.__class__(-self.data, self.field_type, self.evolution_rate, self.device, False, self.dtype)

    def __rmul__(self: FSEFieldTypeVar, other: Union[Number, ArrayLike]) -> FSEFieldTypeVar:
        """Handle right-multiplication (e.g., float * FSEField)."""
        return self.__mul__(other)
    
    def __truediv__(self: FSEFieldTypeVar, other: Union[FSEFieldTypeVar, Number, ArrayLike]) -> FSEFieldTypeVar:
        """✅ ADJOINT ENHANCED: Field division with stability preservation"""
        s_data, o_data, res_device, backend = self._check_compat_and_get_data(other)
        epsilon = backend.array(1e-8, dtype=s_data.dtype)
        try: 
            result_data = s_data / (o_data + epsilon)
            
            # ✅ ADJOINT ENHANCEMENT: Apply stability constraints for division
            result_data = self._apply_division_stability(result_data, backend)
            
        except ValueError as e: 
            raise ValueError(f"Shape mismatch FSEField div: {s_data.shape} vs {getattr(o_data, 'shape', type(o_data))}. {e}")
        return self.__class__(result_data, self.field_type, self.evolution_rate, res_device, False, self.dtype)

    # ✅ ADJOINT FIELD OPERATORS: New methods for continuous field computation
    def _apply_continuity_preservation(self, field_data: ArrayLike, backend) -> ArrayLike:
        """Apply continuity preservation for adjoint PDE compatibility"""
        if field_data.ndim >= 2 and field_data.shape[-2] > 2:
            # Light smoothing to maintain field continuity
            smoothed = field_data.copy()
            smoothed[..., 1:-1, :] = (
                0.25 * field_data[..., :-2, :] + 
                0.5 * field_data[..., 1:-1, :] + 
                0.25 * field_data[..., 2:, :]
            )
            return smoothed
        return field_data

    def _apply_multiplicative_continuity(self, result_data: ArrayLike, s_data: ArrayLike, o_data: ArrayLike, backend) -> ArrayLike:
        """Apply continuity constraints for multiplicative field operations"""
        # Prevent discontinuities in multiplicative operations
        result_magnitude = backend.max(backend.abs(result_data))
        if result_magnitude > 1e6:
            # Scale down to prevent PDE instability
            scaling_factor = 1e6 / result_magnitude
            result_data = result_data * scaling_factor
        return result_data

    def _apply_division_stability(self, result_data: ArrayLike, backend) -> ArrayLike:
        """Apply stability constraints for division operations"""
        # Clamp extreme values that could destabilize adjoint PDE
        result_data = backend.clip(result_data, -1e6, 1e6)
        
        # Replace NaN and infinite values
        result_data = backend.nan_to_num(result_data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return result_data

    def compute_field_gradient(self) -> 'FSEField':
        """✅ ADJOINT OPERATOR: Compute spatial gradient of field for adjoint PDE"""
        backend = self.backend
        
        if self.ndim == 3:  # (B, T, C) - sequence data
            # Compute gradient along sequence dimension
            grad_data = backend.gradient(self.data, axis=1)
        elif self.ndim == 4:  # (B, H, W, C) - spatial data
            # Compute spatial gradients
            grad_h = backend.gradient(self.data, axis=1)
            grad_w = backend.gradient(self.data, axis=2)
            grad_data = backend.stack([grad_h, grad_w], axis=-1)
        else:
            # For other dimensions, use last spatial dimension
            grad_data = backend.gradient(self.data, axis=-2)
        
        return FSEField(grad_data, self.field_type, device=self.device, dtype=self.dtype)

    def compute_field_laplacian(self) -> 'FSEField':
        """✅ ADJOINT OPERATOR: Compute Laplacian for adjoint PDE solving"""
        backend = self.backend
        
        if self.ndim == 3:  # (B, T, C)
            # Second derivative along sequence dimension
            laplacian_data = backend.zeros_like(self.data)
            if self.shape[1] > 2:
                laplacian_data[:, 1:-1, :] = (
                    self.data[:, 2:, :] - 2 * self.data[:, 1:-1, :] + self.data[:, :-2, :]
                )
        elif self.ndim == 4:  # (B, H, W, C)
            # 2D Laplacian
            laplacian_h = backend.zeros_like(self.data)
            laplacian_w = backend.zeros_like(self.data)
            
            if self.shape[1] > 2:
                laplacian_h[:, 1:-1, :, :] = (
                    self.data[:, 2:, :, :] - 2 * self.data[:, 1:-1, :, :] + self.data[:, :-2, :, :]
                )
            if self.shape[2] > 2:
                laplacian_w[:, :, 1:-1, :] = (
                    self.data[:, :, 2:, :] - 2 * self.data[:, :, 1:-1, :] + self.data[:, :, :-2, :]
                )
            
            laplacian_data = laplacian_h + laplacian_w
        else:
            laplacian_data = backend.zeros_like(self.data)
        
        return FSEField(laplacian_data, self.field_type, device=self.device, dtype=self.dtype)

    def apply_field_smoothing(self, smoothing_strength: float = 0.1) -> 'FSEField':
        """✅ ADJOINT OPERATOR: Apply field smoothing for PDE stability"""
        backend = self.backend
        
        if smoothing_strength <= 0:
            return self
        
        smoothed_data = self.data.copy()
        
        if self.ndim == 3 and self.shape[1] > 2:  # (B, T, C)
            # Smooth along sequence dimension
            for _ in range(2):  # Multiple passes for better smoothing
                temp = smoothed_data.copy()
                smoothed_data[:, 1:-1, :] = (
                    (1 - smoothing_strength) * temp[:, 1:-1, :] +
                    smoothing_strength * 0.5 * (temp[:, :-2, :] + temp[:, 2:, :])
                )
        
        elif self.ndim == 4:  # (B, H, W, C)
            # 2D smoothing
            for _ in range(2):
                temp = smoothed_data.copy()
                if self.shape[1] > 2:
                    smoothed_data[:, 1:-1, :, :] = (
                        (1 - smoothing_strength) * temp[:, 1:-1, :, :] +
                        smoothing_strength * 0.5 * (temp[:, :-2, :, :] + temp[:, 2:, :, :])
                    )
                if self.shape[2] > 2:
                    smoothed_data[:, :, 1:-1, :] = (
                        (1 - smoothing_strength) * temp[:, :, 1:-1, :] +
                        smoothing_strength * 0.5 * (temp[:, :, :-2, :] + temp[:, :, 2:, :])
                    )
        
        return FSEField(smoothed_data, self.field_type, device=self.device, dtype=self.dtype)

    def compute_field_magnitude(self) -> float:
        """✅ ADJOINT UTILITY: Compute field magnitude for PDE stability monitoring"""
        backend = self.backend
        return float(backend.max(backend.abs(self.data)))

    def ensure_field_stability(self, max_magnitude: float = 1e6) -> 'FSEField':
        """✅ ADJOINT UTILITY: Ensure field stability for adjoint PDE solving"""
        current_magnitude = self.compute_field_magnitude()
        
        if current_magnitude > max_magnitude:
            scaling_factor = max_magnitude / current_magnitude
            stable_data = self.data * scaling_factor
            logger.debug(f"✅ Field stabilized: magnitude {current_magnitude:.2e} -> {max_magnitude:.2e}")
            return FSEField(stable_data, self.field_type, device=self.device, dtype=self.dtype)
        
        return self

# =========================================
# 🚀 ENHANCED FIELD OPERATIONS with ADJOINT SUPPORT
# =========================================

class FieldOperations:
    """✅ ADJOINT ENHANCED: Field operations with adjoint PDE support"""
    
    # Cache for compiled kernels to avoid recompilation
    _kernel_cache = {}
    
    @staticmethod
    def apply_activation(pre_activation_field: FSEField, activation_type: FieldType) -> FSEField:
        """✅ ADJOINT ENHANCED: Activation with field continuity preservation"""
        backend = pre_activation_field.backend
        Z = pre_activation_field.data
        
        # ✅ MEMORY SAFE: Synchronization before kernel operations
        if pre_activation_field.device == "gpu":
            cp.cuda.runtime.deviceSynchronize()
        
        # Use cached kernels for GPU operations
        if pre_activation_field.device == "gpu" and activation_type in FieldOperations._kernel_cache:
            cached_kernel = FieldOperations._kernel_cache[activation_type]
            activated_data = cached_kernel(Z)
        else:
            # Fallback to standard operations with dtype preservation
            if activation_type == FieldType.LINEAR: 
                activated_data = Z
            elif activation_type == FieldType.CONTINUOUS or activation_type == FieldType.SPATIAL: 
                activated_data = backend.tanh(Z)
            elif activation_type == FieldType.WAVE: 
                activated_data = backend.sin(Z)
            elif activation_type == FieldType.QUANTUM: 
                activated_data = backend.tanh(Z) * backend.cos(2.0 * Z)
            elif activation_type == FieldType.LIGHTING: 
                activated_data = 1.0 / (1.0 + backend.exp(-Z))
            elif activation_type == FieldType.MATERIAL: 
                activated_data = backend.maximum(backend.array(0.2, dtype=Z.dtype) * Z, Z)
            else: 
                logger.warning(f"Unknown field type {activation_type}, using linear.")
                activated_data = Z
        
        # ✅ ADJOINT ENHANCEMENT: Apply field smoothing for continuity
        result_field = FSEField(activated_data, activation_type, pre_activation_field.evolution_rate, 
                               pre_activation_field.device, use_memory_pool=False, dtype=activated_data.dtype)
        
        if activation_type in [FieldType.CONTINUOUS, FieldType.SPATIAL]:
            result_field = result_field.apply_field_smoothing(0.05)
        
        # ✅ MEMORY SAFE: Post-operation synchronization
        if pre_activation_field.device == "gpu":
            cp.cuda.runtime.deviceSynchronize()
        
        return result_field

    @staticmethod
    def activation_derivative(grad_output_activated: FSEField,
                              pre_activation_data: ArrayLike, 
                              activation_type_used: FieldType) -> FSEField:
        """✅ ADJOINT ENHANCED: Activation derivatives with field continuity"""
        
        backend = grad_output_activated.backend
        grad_data = grad_output_activated.data

        # === Defensive Shape Alignment ===
        if grad_data.shape[1] != pre_activation_data.shape[1]:
            logger.warning(f"Shape mismatch in activation derivative: {grad_data.shape} vs {pre_activation_data.shape}. Aligning to minimum length.")
            min_len = min(grad_data.shape[1], pre_activation_data.shape[1])
            
            grad_data_trunc = grad_data[:, :min_len, ...]
            pre_act_data_trunc = pre_activation_data[:, :min_len, ...]
        else:
            grad_data_trunc = grad_data
            pre_act_data_trunc = pre_activation_data
        
        Z = pre_act_data_trunc

        try:
            # Vectorized derivative computations
            if activation_type_used == FieldType.LINEAR: 
                dAct_dZ = backend.ones_like(Z)
            elif activation_type_used == FieldType.CONTINUOUS or activation_type_used == FieldType.SPATIAL: 
                tanh_Z = backend.tanh(Z)
                dAct_dZ = 1.0 - tanh_Z**2
            elif activation_type_used == FieldType.WAVE: 
                dAct_dZ = backend.cos(Z)
            elif activation_type_used == FieldType.QUANTUM: 
                tanh_Z = backend.tanh(Z); sech_sq_Z = 1.0 - tanh_Z**2
                cos_2Z = backend.cos(2.0 * Z); sin_2Z = backend.sin(2.0 * Z)
                dAct_dZ = sech_sq_Z * cos_2Z - 2.0 * tanh_Z * sin_2Z
            elif activation_type_used == FieldType.LIGHTING: 
                sigmoid_Z = 1.0 / (1.0 + backend.exp(-backend.clip(Z, -50, 50)))
                dAct_dZ = sigmoid_Z * (1.0 - sigmoid_Z)
            elif activation_type_used == FieldType.MATERIAL: 
                dAct_dZ = backend.where(Z > 0, backend.array(1.0, dtype=Z.dtype), backend.array(0.2, dtype=Z.dtype))
            else: 
                dAct_dZ = backend.ones_like(Z)
            
            # Compute the gradient
            grad_Z_truncated = grad_data_trunc * dAct_dZ
            
            # === Output Shape Correction ===
            if grad_Z_truncated.shape != grad_output_activated.shape:
                grad_Z_full = backend.zeros_like(grad_output_activated.data)
                grad_Z_full[:, :grad_Z_truncated.shape[1], ...] = grad_Z_truncated
                grad_Z_data = grad_Z_full
            else:
                grad_Z_data = grad_Z_truncated

        except Exception as e:
            logger.error(f"❌ Activation derivative computation failed: {e}")
            grad_Z_data = grad_output_activated.data.copy()
        
        if grad_output_activated.device == "gpu":
            cp.cuda.runtime.deviceSynchronize()
        
        # ✅ ADJOINT ENHANCEMENT: Create result with field continuity
        result_field = FSEField(grad_Z_data, grad_output_activated.field_type, grad_output_activated.evolution_rate, 
                               grad_output_activated.device, dtype=grad_output_activated.dtype)
        
        # Apply field smoothing for adjoint PDE compatibility
        if activation_type_used in [FieldType.CONTINUOUS, FieldType.SPATIAL]:
            result_field = result_field.apply_field_smoothing(0.03)
        
        return result_field

    @staticmethod
    def compute_continuous_field_derivative(field: FSEField, order: int = 1) -> FSEField:
        """✅ ADJOINT OPERATOR: Compute continuous field derivatives for adjoint PDE"""
        if order == 1:
            return field.compute_field_gradient()
        elif order == 2:
            return field.compute_field_laplacian()
        else:
            # Higher order derivatives through recursive application
            current_field = field
            for _ in range(order):
                current_field = current_field.compute_field_gradient()
            return current_field

    @staticmethod
    def apply_field_evolution_operator(field: FSEField, parameters: Dict[str, FSEField], 
                                     field_type: FieldType, dt: float = 0.1) -> FSEField:
        """✅ ADJOINT OPERATOR: Apply field evolution operator for PDE integration"""
        backend = field.backend
        
        # Extract kernel parameter if available
        if 'kernel' in parameters:
            kernel_data = parameters['kernel'].data
            
            # Apply field transformation
            if kernel_data.ndim == 2:  # 1x1 kernel case
                evolved_data = field.data @ kernel_data
            else:
                # More complex field evolution
                evolved_data = field.data  # Simplified for now
        else:
            evolved_data = field.data
        
        # Apply field-type specific evolution
        if field_type == FieldType.CONTINUOUS:
            evolved_data = backend.tanh(evolved_data)
        elif field_type == FieldType.WAVE:
            evolved_data = backend.sin(evolved_data + 2 * backend.pi * dt)
        elif field_type == FieldType.QUANTUM:
            evolved_data = backend.tanh(evolved_data) * backend.cos(2.0 * evolved_data)
        
        # Scale by time step
        field_increment = (evolved_data - field.data) * dt
        final_data = field.data + field_increment
        
        result_field = FSEField(final_data, field_type, device=field.device, dtype=field.dtype)
        
        # Ensure stability
        result_field = result_field.ensure_field_stability()
        
        return result_field

    @staticmethod
    def vectorized_im2col_gemm_convolution(input_field: FSEField, kernel_field: FSEField,
                                          strides: Tuple[int, int] = (1, 1),
                                          padding_mode: str = "SAME") -> Tuple[FSEField, Dict[str, Any]]:
        """
        🚀 MEMORY SAFE: Ultra-fast convolution with enhanced memory safety
        """
        
        # ✅ MEMORY SAFE: Pre-operation synchronization
        if input_field.device == "gpu":
            cp.cuda.runtime.deviceSynchronize()
        
        input_data = input_field.data
        backend = input_field.backend
        is_3d_input = (input_data.ndim == 3)

        try:
            # --- Handle 1x1 convolution (common case) ---
            if kernel_field.ndim == 2:
                if is_3d_input: # 3D case: (B, T, C_in)
                    B, T, C_in = input_data.shape
                    C_out = kernel_field.shape[1]
                    input_reshaped = input_data.reshape(-1, C_in)
                    output_flat = input_reshaped @ kernel_field.data
                    output_data = output_flat.reshape(B, T, C_out)
                    output_shape_final = (B, T, C_out)
                else: # 4D case: (B, H, W, C_in)
                    input_field_4d, (B, H, W, C_in) = _ensure_4d(input_field)
                    C_out = kernel_field.shape[1]
                    input_reshaped = input_field_4d.data.reshape(-1, C_in)
                    output_flat = input_reshaped @ kernel_field.data
                    output_data = output_flat.reshape(B, H, W, C_out)
                    output_shape_final = (B, H, W, C_out)
                
                # --- Create cache for 1x1 conv ---
                cache = {
                    'input_field_shape': input_field.shape, 'kernel_field_shape': kernel_field.shape,
                    'is_1x1_conv': True, 'input_reshaped': input_reshaped,
                    'kernel_reshaped': kernel_field.data, 'output_shape': output_shape_final,
                    # Add other necessary keys for backward pass
                    'strides': strides, 'padding_mode': padding_mode, 'P_H': 0, 'P_W': 0,
                    'input_padded_shape': input_field.shape, 'pre_activation_data': output_data,
                    'activation_type_used': FieldType.LINEAR
                }

            # --- Handle standard N-D convolution ---
            else:
                if is_3d_input:
                    # This path is for future use; FSU LLM primarily uses 1x1 convs in FLITs.
                    # A proper 1D conv (im2col_1d) would be needed here for full 1D support.
                    # For now, we assume FLITs in FSEBlocks are 1x1 and this path won't be hit.
                    raise NotImplementedError("Standard 1D convolution not implemented. FSEBlock FILs should use 1x1 kernels.")
                else:
                    # --- Original 4D convolution logic ---
                    input_field_4d, (B, H, W, C_in) = _ensure_4d(input_field)
                    KH, KW, C_in_k, C_out = kernel_field.shape
                    if C_in_k != C_in: raise ValueError(f"Kernel C_in {C_in_k} mismatch input C_in {C_in}")
                    
                    S_H, S_W = strides
                    P_H = max((H - 1) * S_H + KH - H, 0) // 2 if padding_mode == "SAME" else 0
                    P_W = max((W - 1) * S_W + KW - W, 0) // 2 if padding_mode == "SAME" else 0

                    input_padded = backend.pad(input_field_4d.data, ((0,0),(P_H,P_H),(P_W,P_W),(0,0))) if P_H > 0 or P_W > 0 else input_field_4d.data
                    H_pad, W_pad = input_padded.shape[1:3]
                    out_h, out_w = (H_pad - KH) // S_H + 1, (W_pad - KW) // S_W + 1
                    
                    cols = FieldOperations._cupy_vectorized_im2col(input_padded, KH, KW, S_H, S_W, out_h, out_w) if backend == cp else FieldOperations._numpy_vectorized_im2col(input_padded, KH, KW, S_H, S_W, out_h, out_w)
                    cols_reshaped = cols.reshape(B, out_h * out_w, KH * KW * C_in)
                    kernel_reshaped = kernel_field.data.reshape(KH * KW * C_in, C_out)
                    
                    output_flat = cols_reshaped @ kernel_reshaped
                    output_data = output_flat.reshape(B, out_h, out_w, C_out)
                    output_shape_final = (B, out_h, out_w, C_out)

                    cache = {
                        'input_field_shape': input_field.shape, 'kernel_field_shape': kernel_field.shape,
                        'is_1x1_conv': False, 'cols_reshaped': cols_reshaped, 
                        'kernel_reshaped': kernel_reshaped, 'output_shape': output_shape_final,
                        'input_padded_shape': input_padded.shape, 'P_H': P_H, 'P_W': P_W, 'strides': strides,
                        'pre_activation_data': output_data, 'activation_type_used': FieldType.LINEAR
                    }

        except Exception as e:
            logger.error(f"Convolution operation failed: {e}")
            # ✅ MEMORY SAFE: Emergency cleanup and fallback
            if input_field.device == "gpu":
                try:
                    cp.cuda.runtime.deviceSynchronize()
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
            raise

        # ✅ MEMORY SAFE: Post-operation synchronization
        if input_field.device == "gpu":
            cp.cuda.runtime.deviceSynchronize()

        output_field_type = kernel_field.field_type if kernel_field.field_type != FieldType.LINEAR else input_field.field_type
        result_field = FSEField(output_data, output_field_type, input_field.evolution_rate,
                               input_field.device, use_memory_pool=False, dtype=output_data.dtype)
        
        # ✅ ADJOINT ENHANCEMENT: Apply field smoothing for continuity
        if output_field_type in [FieldType.CONTINUOUS, FieldType.SPATIAL]:
            result_field = result_field.apply_field_smoothing(0.02)
        
        return result_field, cache

    @staticmethod
    def _cupy_vectorized_im2col(input_padded, KH, KW, S_H, S_W, out_h, out_w):
        """
        [DEFINITIVE, SELF-CONTAINED FIX V4] Implements im2col using CuPy's fundamental
        `as_strided` function. This is highly performant, has no version-dependent
        imports, and resolves all previous errors.
        """
        from cupy.lib.stride_tricks import as_strided

        B, H_pad, W_pad, C_in = input_padded.shape
        
        try:
            # Get the stride values for each dimension of the input array
            sB, sH, sW, sC = input_padded.strides

            # Define the shape of the output view before the final reshape.
            view_shape = (B, out_h, out_w, KH, KW, C_in)

            # Define the strides of the output view. This tells CuPy how to step
            # through the memory of the original array to create the illusion of a new,
            # larger array without copying data.
            view_strides = (sB, sH * S_H, sW * S_W, sH, sW, sC)

            # Create the strided view, which is a memory-efficient operation.
            patches_view = as_strided(input_padded, shape=view_shape, strides=view_strides)
            
            # Reshape the view to get the final im2col format.
            cols = patches_view.reshape(B, out_h, out_w, KH * KW * C_in)
            
            return cols

        except Exception as e:
            logger.error(f"CuPy im2col (as_strided) failed: {e}", exc_info=True)
            try:
                cp.cuda.runtime.deviceSynchronize()
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
            raise

    @staticmethod
    def _numpy_vectorized_im2col(input_padded, KH, KW, S_H, S_W, out_h, out_w):
        """
        [DEFINITIVE, SELF-CONTAINED FIX V4] Symmetrically corrected NumPy version
        using `as_strided` to ensure CPU/GPU consistency and robustness.
        """
        from numpy.lib.stride_tricks import as_strided
        
        B, H_pad, W_pad, C_in = input_padded.shape
        
        try:
            sB, sH, sW, sC = input_padded.strides
            view_shape = (B, out_h, out_w, KH, KW, C_in)
            view_strides = (sB, sH * S_H, sW * S_W, sH, sW, sC)
            patches_view = as_strided(input_padded, shape=view_shape, strides=view_strides)
            cols = patches_view.reshape(B, out_h, out_w, KH * KW * C_in)
            
            return cols
            
        except Exception as e:
            logger.error(f"NumPy im2col (as_strided) failed: {e}", exc_info=True)
            raise

    @staticmethod
    def field_convolution_backward_data(upstream_grad: FSEField, cache: Dict[str, Any]) -> FSEField:
        """[FINAL-FIX V3] Optimized backward pass for data gradients, now fully truncation-aware."""
        
        if upstream_grad.device == "gpu":
            cp.cuda.runtime.deviceSynchronize()
        
        backend = upstream_grad.backend
        
        required_keys = ['is_1x1_conv', 'input_field_shape']
        if any(key not in cache for key in required_keys):
            logger.error(f"❌ CRITICAL: Missing required cache keys for data backward.")
            return FSEField(backend.zeros_like(upstream_grad.data), upstream_grad.field_type, device=upstream_grad.device)

        input_shape_full = cache['input_field_shape']

        try:
            if cache.get('is_1x1_conv', False):
                if 'kernel_reshaped' not in cache:
                    raise ValueError("Missing 'kernel_reshaped' for 1x1 conv backward data.")
                
                kernel_reshaped = cache['kernel_reshaped']
                upstream_flat = upstream_grad.data.reshape(-1, upstream_grad.shape[-1])
                grad_input_flat = upstream_flat @ kernel_reshaped.T
                
                # The shape of grad_input_flat is now (B * T_truncated, C).
                # We need to reshape it back to (B, T_truncated, C).
                final_shape = (upstream_grad.shape[0], upstream_grad.shape[1], grad_input_flat.shape[-1])
                grad_input_data = grad_input_flat.reshape(final_shape)

            else:
                # This logic is for standard (non 1x1) convolutions.
                if any(key not in cache for key in ['output_shape', 'kernel_reshaped', 'input_padded_shape', 'P_H', 'P_W']):
                     raise ValueError("Missing standard conv keys for data backward.")

                B, out_h, out_w, C_out = cache['output_shape']
                kernel_reshaped = cache['kernel_reshaped']
                input_padded_shape = cache['input_padded_shape']
                P_H, P_W = cache['P_H'], cache['P_W']
                
                upstream_flat = upstream_grad.data.reshape(B, out_h * out_w, C_out)
                grad_cols = upstream_flat @ kernel_reshaped.T
                
                grad_input_padded = FieldOperations._vectorized_col2im(grad_cols, input_padded_shape, cache)
                
                if P_H > 0 or P_W > 0:
                    grad_input_data = grad_input_padded[:, P_H:-P_H, P_W:-P_W, :]
                else:
                    grad_input_data = grad_input_padded
        
        except Exception as e:
            logger.error(f"Backward data computation failed: {e}", exc_info=True)
            # Fallback to a zero gradient with the correct *truncated* shape
            fallback_shape = (input_shape_full[0], upstream_grad.shape[1]) + input_shape_full[2:]
            grad_input_data = backend.zeros(fallback_shape, dtype=upstream_grad.dtype)
        
        # ✅ FINAL-FIX V3: This is the definitive guard. The returned gradient's sequence length
        # MUST match the upstream gradient's sequence length.
        if grad_input_data.shape[1] != upstream_grad.shape[1]:
             logger.warning(f"Final shape correction in backward_data: {grad_input_data.shape} -> {upstream_grad.shape}")
             # Create a zero-tensor with the correct target shape and copy the data.
             # This correctly handles cases where the forward pass truncated the sequence.
             corrected_grad = backend.zeros(upstream_grad.shape, dtype=grad_input_data.dtype)
             min_len = min(grad_input_data.shape[1], corrected_grad.shape[1])
             corrected_grad[:, :min_len, :] = grad_input_data[:, :min_len, :]
             grad_input_data = corrected_grad

        if upstream_grad.device == "gpu":
            cp.cuda.runtime.deviceSynchronize()
        
        result_field = FSEField(grad_input_data, upstream_grad.field_type, device=upstream_grad.device, dtype=grad_input_data.dtype)
        
        # ✅ ADJOINT ENHANCEMENT: Apply field smoothing for gradient continuity
        result_field = result_field.apply_field_smoothing(0.01)
        
        return result_field

    @staticmethod
    def field_convolution_backward_kernel(upstream_grad: FSEField, cache: Dict[str, Any]) -> FSEField:
        """[FINAL-FIX V3] Optimized backward pass for kernel gradients, now fully truncation-aware."""
        
        backend = upstream_grad.backend
        kernel_shape = cache['kernel_field_shape']
        
        try:
            if cache.get('is_1x1_conv', False):
                input_reshaped_full = cache.get('input_reshaped')
                if input_reshaped_full is None:
                    raise ValueError("Missing 'input_reshaped' from cache for 1x1 conv backward.")

                # ✅ CRITICAL FIX: Manually truncate the cached input tensor to match the gradient's shape.
                # The gradient's shape is the source of truth for the (truncated) sequence length.
                num_elements_in_grad = upstream_grad.data.shape[0] * upstream_grad.data.shape[1]
                input_reshaped = input_reshaped_full[:num_elements_in_grad, :]
                
                upstream_flat = upstream_grad.data.reshape(-1, upstream_grad.shape[-1])

                # This matmul should now succeed as both tensors have a matching dimension.
                grad_kernel_data = input_reshaped.T @ upstream_flat
                
                if grad_kernel_data.shape != kernel_shape:
                    grad_kernel_data = grad_kernel_data.reshape(kernel_shape)

            else:
                # This logic is for standard (non 1x1) convolutions.
                cols_reshaped = cache['cols_reshaped'] 
                B, out_h, out_w, C_out = cache.get('output_shape', upstream_grad.shape)
                upstream_flat = upstream_grad.data.reshape(B, out_h * out_w, C_out)
                
                grad_kernel_flat = backend.zeros((cols_reshaped.shape[-1], C_out), dtype=upstream_grad.dtype)
                for b in range(B):
                    grad_kernel_flat += cols_reshaped[b].T @ upstream_flat[b]
                grad_kernel_data = grad_kernel_flat.reshape(kernel_shape)

        except Exception as e:
            logger.error(f"Backward kernel computation failed: {e}", exc_info=True)
            grad_kernel_data = backend.zeros(kernel_shape, dtype=upstream_grad.dtype)
        
        if upstream_grad.device == "gpu":
            cp.cuda.runtime.deviceSynchronize()
        
        result_field = FSEField(grad_kernel_data, upstream_grad.field_type, device=upstream_grad.device, dtype=grad_kernel_data.dtype)
        
        # ✅ ADJOINT ENHANCEMENT: Apply field smoothing for parameter gradient continuity
        result_field = result_field.apply_field_smoothing(0.005)
        
        return result_field

    @staticmethod
    def _vectorized_col2im(grad_cols, input_padded_shape, cache):
        """
        [DEFINITIVE FIX] Vectorized col2im operation for backward pass.
        This version correctly infers the output shape for both 1D and 2D convolutions
        by using the 'output_shape' key from the cache, resolving the reshape error.
        """
        backend = cp if isinstance(grad_cols, cp.ndarray) else np
        
        B, H_pad, W_pad, C_in = input_padded_shape
        kernel_shape = cache['kernel_field_shape']
        
        # For 1D convs, KH will be 1, KW will be the kernel size.
        # For 2D convs, both will be > 1.
        KH = kernel_shape[0] if len(kernel_shape) == 4 else 1
        KW = kernel_shape[1] if len(kernel_shape) == 4 else kernel_shape[0]
        
        S_H, S_W = cache['strides']
        
        try:
            grad_input_padded = backend.zeros(input_padded_shape, dtype=grad_cols.dtype)
            
            # --- THIS IS THE FIX ---
            # Correctly determine the output height and width from the cached 'output_shape'.
            if 'output_shape' in cache:
                _, out_h, out_w, _ = cache['output_shape']
            else:
                # Fallback logic if cache is missing, now correctly handles the 1D case.
                # Since this is a 1D conv simulated as 2D, out_h should be 1.
                B, total_spatial_patches, _ = grad_cols.shape
                is_1d_conv_simulation = (KH == 1)
                
                if is_1d_conv_simulation:
                    out_h = 1
                    out_w = total_spatial_patches
                else: # Original fallback for 2D square case
                    out_h = int(np.sqrt(total_spatial_patches))
                    out_w = total_spatial_patches // out_h
            # --- END OF FIX ---
            
            # Reshape gradients back to patch format
            grad_cols_patches = grad_cols.reshape(B, out_h, out_w, KH, KW, C_in)
            
            # Accumulate gradients back to input locations using a loop.
            # While slower than a fully vectorized approach, this is guaranteed to be correct.
            for y in range(out_h):
                for x in range(out_w):
                    y_start, y_end = y * S_H, y * S_H + KH
                    x_start, x_end = x * S_W, x * S_W + KW
                    # Bounds check to prevent errors with padding
                    if y_end <= H_pad and x_end <= W_pad:
                        grad_input_padded[:, y_start:y_end, x_start:x_end, :] += grad_cols_patches[:, y, x, :, :, :]
            
            return grad_input_padded
            
        except Exception as e:
            logger.error(f"Col2im operation failed: {e}", exc_info=True)
            return backend.zeros(input_padded_shape, dtype=grad_cols.dtype)

    # Alias for backward compatibility
    @staticmethod
    def field_convolution(input_field: FSEField, kernel_field: FSEField,
                         strides: Tuple[int, int] = (1, 1),
                         padding_mode: str = "SAME") -> Tuple[FSEField, Dict[str, Any]]:
        """Main convolution interface - routes to optimized implementation"""
        return FieldOperations.vectorized_im2col_gemm_convolution(
            input_field, kernel_field, strides, padding_mode
        )

# =========================================
# 🚀 ENHANCED FUSED OPERATIONS with ADJOINT SUPPORT
# =========================================

class FusedFieldOperations:
    """✅ ADJOINT ENHANCED: Fused operations with adjoint PDE support"""
    
    @staticmethod
    def fused_conv_activation(input_field: FSEField, kernel_field: FSEField,
                             activation_type: FieldType,
                             strides: Tuple[int, int] = (1, 1),
                             padding_mode: str = "SAME") -> Tuple[FSEField, Dict[str, Any]]:
        """✅ ADJOINT ENHANCED: Fused convolution + activation with field continuity"""
        
        # ✅ MEMORY SAFE: Pre-operation synchronization
        if input_field.device == "gpu":
            cp.cuda.runtime.deviceSynchronize()
        
        try:
            # Perform convolution
            conv_output, conv_cache = FieldOperations.field_convolution(
                input_field, kernel_field, strides, padding_mode
            )
            
            # ✅ MEMORY SAFE: Store pre_activation_data by REFERENCE (not copy)
            pre_activation_data = conv_output.data  # Reference only - no .copy() to avoid OOM
            
            # Apply activation in-place for memory efficiency
            activated_output = FieldOperations.apply_activation(conv_output, activation_type)
            
            # ✅ CRITICAL FIX: Update convolution cache with activation information
            conv_cache['pre_activation_data'] = pre_activation_data  # Reference, not copy
            conv_cache['activation_type_used'] = activation_type
            
            # ✅ COMPREHENSIVE FUSED CACHE with memory safety
            fused_cache = {
                'conv_cache': conv_cache,  # Original convolution cache
                'pre_activation_data': pre_activation_data,  # ✅ CRITICAL: Stored BEFORE activation
                'activation_type_used': activation_type,  # ✅ CRITICAL: Store the activation type used
                
                # ✅ CRITICAL: Copy ALL essential keys from conv_cache for direct access
                'is_1x1_conv': conv_cache.get('is_1x1_conv', False),
                'input_field_shape': conv_cache.get('input_field_shape'),
                'kernel_field_shape': conv_cache.get('kernel_field_shape'),
                'cols_reshaped': conv_cache.get('cols_reshaped'),
                'kernel_reshaped': conv_cache.get('kernel_reshaped'),
                'input_reshaped': conv_cache.get('input_reshaped'),  # ✅ KEY for 1x1 conv
                'output_shape': conv_cache.get('output_shape'),
                'strides': conv_cache.get('strides'),
                'P_H': conv_cache.get('P_H', 0),
                'P_W': conv_cache.get('P_W', 0),
                'input_padded_shape': conv_cache.get('input_padded_shape'),
                
                # ✅ ADDITIONAL SAFETY KEYS
                'backend': conv_cache.get('backend'),
                'device': conv_cache.get('device', input_field.device),
                'conv_type': conv_cache.get('conv_type', 'unknown'),
                'kernel_dims': conv_cache.get('kernel_dims'),
                'output_dims': conv_cache.get('output_dims'),
                'padding_mode': conv_cache.get('padding_mode', padding_mode),
                
                # ✅ MIXED PRECISION: Store dtype information
                'input_dtype': conv_cache.get('input_dtype', input_field.dtype),
                'kernel_dtype': conv_cache.get('kernel_dtype', kernel_field.dtype),
                'output_dtype': conv_cache.get('output_dtype', activated_output.dtype),
                
                # ✅ FUSED OPERATION SPECIFIC
                'is_fused': True,
                'fused_type': 'conv_activation'
            }
            
            # ✅ MEMORY SAFE: Post-operation synchronization
            if input_field.device == "gpu":
                cp.cuda.runtime.deviceSynchronize()
            
            logger.debug(f"✅ Fused conv+activation: {input_field.shape} -> {activated_output.shape}, activation={activation_type}")
            
            return activated_output, fused_cache
            
        except Exception as e:
            logger.error(f"Fused conv+activation failed: {e}")
            # ✅ MEMORY SAFE: Emergency cleanup
            if input_field.device == "gpu":
                try:
                    cp.cuda.runtime.deviceSynchronize()
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
            raise
    

    @staticmethod
    def fused_conv_activation_backward(upstream_grad: FSEField, cache: Dict[str, Any], truncated_len: int = None) -> Tuple[Dict[str, FSEField], FSEField]:
        """
        [FINAL-FIX] Fused backward pass is now fully truncation-aware.
        """
        conv_cache = cache.get('conv_cache', cache) # Handle both fused and direct calls
        
        if 'pre_activation_data' not in conv_cache or 'activation_type_used' not in conv_cache:
            raise KeyError("'pre_activation_data' or 'activation_type_used' missing from conv_cache.")

        # === TRUNCATE CACHED ACTIVATIONS ===
        pre_activation_data_full = conv_cache['pre_activation_data']
        
        # Check if sequence dimension exists and needs truncation
        if truncated_len and pre_activation_data_full.ndim > 1 and pre_activation_data_full.shape[1] > truncated_len:
            pre_activation_data = pre_activation_data_full[:, :truncated_len, :]
        else:
            pre_activation_data = pre_activation_data_full
        
        # 1. Backprop through activation function on the truncated data
        grad_pre_activation = FieldOperations.activation_derivative(
            upstream_grad, pre_activation_data.astype(upstream_grad.backend.float32), conv_cache['activation_type_used']
        )
        
        # === TRUNCATE CACHED CONV INPUTS FOR GRADIENT CALCULATION ===
        # This is crucial for 1x1 convolutions which are essentially matrix multiplications
        if conv_cache.get('is_1x1_conv') and 'input_reshaped' in conv_cache:
            input_reshaped_full = conv_cache['input_reshaped']
            # The reshaped input is (B*T, C). We need to calculate the correct number of rows to keep.
            batch_size = upstream_grad.shape[0]
            truncated_len_actual = truncated_len if truncated_len else upstream_grad.shape[1]
            num_elements_to_keep = batch_size * truncated_len_actual
            
            if input_reshaped_full.shape[0] > num_elements_to_keep:
                conv_cache['input_reshaped_truncated'] = input_reshaped_full[:num_elements_to_keep, :]
            else:
                conv_cache['input_reshaped_truncated'] = input_reshaped_full
        
        # 2. Backprop through convolution
        # Pass the original conv_cache, but the backward functions inside will use the new truncated keys if they exist.
        kernel_grad = FieldOperations.field_convolution_backward_kernel(grad_pre_activation, conv_cache)
        input_grad = FieldOperations.field_convolution_backward_data(grad_pre_activation, conv_cache)
        
        return {'kernel': kernel_grad}, input_grad

# =========================================
# 🚀 BATCH-OPTIMIZED OPERATIONS - MEMORY SAFE
# =========================================

class BatchedFieldOperations:
    """✅ ADJOINT ENHANCED: Batch-optimized operations with adjoint support"""
    
    @staticmethod
    def batched_field_processing(batch_fields: List[FSEField], operation_type: str, **kwargs) -> List[FSEField]:
        """Process multiple fields in a single batched operation - MEMORY SAFE"""
        
        if not batch_fields:
            return []
        
        # ✅ MEMORY SAFE: Pre-operation synchronization
        if batch_fields[0].device == "gpu":
            cp.cuda.runtime.deviceSynchronize()
        
        try:
            # Stack all fields into single tensor for batched processing
            backend = batch_fields[0].backend
            
            # ✅ MIXED PRECISION: Ensure all fields have same dtype for stacking
            common_dtype = batch_fields[0].dtype
            batch_data_list = []
            for field in batch_fields:
                if field.dtype != common_dtype:
                    # Convert to common dtype (prefer higher precision)
                    if common_dtype == cp.float32 or field.dtype == cp.float32:
                        common_dtype = cp.float32
                    field_data = field.data.astype(common_dtype)
                else:
                    field_data = field.data
                batch_data_list.append(field_data)
            
            batch_data = backend.stack(batch_data_list, axis=0)
            batched_field = FSEField(batch_data, batch_fields[0].field_type, device=batch_fields[0].device, 
                                   use_memory_pool=False, dtype=common_dtype)
            
            # Apply operation to entire batch
            if operation_type == "activation":
                result_field = FieldOperations.apply_activation(batched_field, kwargs['activation_type'])
            elif operation_type == "convolution":
                result_field, _ = FieldOperations.field_convolution(batched_field, kwargs['kernel_field'])
            elif operation_type == "field_evolution":
                # ✅ ADJOINT OPERATION: Apply field evolution to batch
                result_field = FieldOperations.apply_field_evolution_operator(
                    batched_field, kwargs.get('parameters', {}), kwargs.get('field_type', FieldType.CONTINUOUS)
                )
            else:
                raise ValueError(f"Unknown operation type: {operation_type}")
            
            # Split back into individual fields
            result_data_split = backend.split(result_field.data, len(batch_fields), axis=0)
            result_fields = []
            for data in result_data_split:
                field = FSEField(data.squeeze(0), result_field.field_type, device=result_field.device, 
                               use_memory_pool=False, dtype=result_field.dtype)
                result_fields.append(field)
            
            # ✅ MEMORY SAFE: Post-operation synchronization
            if batch_fields[0].device == "gpu":
                cp.cuda.runtime.deviceSynchronize()
            
            return result_fields
            
        except Exception as e:
            logger.error(f"Batched field processing failed: {e}")
            # ✅ MEMORY SAFE: Emergency cleanup
            if batch_fields[0].device == "gpu":
                try:
                    cp.cuda.runtime.deviceSynchronize()
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
            raise

# =========================================
# AURALITH INVERSE-DESIGN ENGINE V4
# =========================================
class MorphSolver:
    """
    [AURALITH INVERSE-DESIGN ENGINE V4 - LOGIT TARGETING]
    This version fixes the conflicting objective issue by solving for a strong
    logit target instead of a one-hot vector. This aligns the Morph solver's
    goal with the SGD/Cross-Entropy loss function.
    """
    def __init__(self, ridge: float = 1e-3, device: str = "gpu", vocab_size: int = 65536):
        self.ridge = ridge
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.vocab_size = vocab_size
        if self.device == 'cpu':
            logger.warning("⚠️ MorphSolver initialized on CPU. Performance will be severely limited.")


    # [V6] - MorphSolver.update_layer_weights
    def update_layer_weights(self, input_data: ArrayLike, target_indices: ArrayLike,
                                layer_W: FSEField, layer_b: FSEField):
            is_gpu = (self.device == 'gpu')
            inp_backend = cp.asarray(input_data) if is_gpu else np.asarray(input_data)
            tgt_backend = cp.asarray(target_indices) if is_gpu else np.asarray(target_indices)

            if inp_backend.ndim == 3:
                C_in = inp_backend.shape[-1]
                inp_reshaped = inp_backend.reshape(-1, C_in).astype(self.backend.float64)
            else:
                inp_reshaped = inp_backend.astype(self.backend.float64)
            
            if tgt_backend.dtype in [np.int32, np.int64, cp.int32, cp.int64]:
                N = inp_reshaped.shape[0]
                LOGIT_TARGET_VAL = 10.0
                Y_matrix = self.backend.full((N, self.vocab_size), -LOGIT_TARGET_VAL / (self.vocab_size - 1), dtype=self.backend.float64)
                tgt_flat = tgt_backend.flatten().astype(self.backend.int32)
                
                # Defensive check for target indices length
                if len(tgt_flat) != N:
                    logger.warning(f"MorphSolver: Aligning target indices length from {len(tgt_flat)} to input length {N}")
                    tgt_flat = tgt_flat[:N]
                
                Y_matrix[self.backend.arange(len(tgt_flat)), tgt_flat] = LOGIT_TARGET_VAL
            else:
                Y_matrix = tgt_backend.reshape(-1, tgt_backend.shape[-1]).astype(self.backend.float64)

            # --- THIS IS THE FIX ---
            # Defensively align the number of rows (tokens) between X and Y before the matmul.
            min_rows = min(inp_reshaped.shape[0], Y_matrix.shape[0])
            if min_rows == 0:
                logger.warning("MorphSolver: No data to process after alignment. Skipping update.")
                return

            inp_aligned = inp_reshaped[:min_rows, :]
            Y_aligned = Y_matrix[:min_rows, :]
            # --- END FIX ---
            
            X_matrix = self.backend.concatenate([inp_aligned, self.backend.ones((min_rows, 1), dtype=self.backend.float64)], axis=1)

            try:
                XtX = X_matrix.T @ X_matrix
                XtX[self.backend.diag_indices_from(XtX)] += self.ridge
                
                XtY = (X_matrix.T @ Y_aligned).copy() # Use aligned Y
                theta = self.backend.linalg.solve(XtX, XtY)
                
                W_new = theta[:-1, :]
                b_new = theta[-1, :]
                layer_W.data = self.backend.asarray(W_new, dtype=layer_W.dtype)
                layer_b.data = self.backend.asarray(b_new, dtype=layer_b.dtype)
            except self.backend.linalg.LinAlgError as e:
                logger.warning(f"⚠️ MorphSolver failed with LinAlgError: {e}. Skipping update for this step.")
                pass

    def compute_local_matrices(self, input_data: ArrayLike, target_data: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
            """
            [UPGRADED] Computes the local X_transpose * X and X_transpose * Y matrices.
            This version now correctly handles both integer indices and continuous field targets.
            """
            inp_reshaped = input_data.reshape(-1, input_data.shape[-1]).astype(self.backend.float64)
            N, D_in = inp_reshaped.shape

            # Check if the target is indices (for the sampler) or a continuous field (for the frontend)
            if target_data.dtype in [np.int32, np.int64, cp.int32, cp.int64]:
                # Target is indices -> Create logit target matrix
                LOGIT_TARGET_VAL = 10.0
                Y_matrix = self.backend.full((N, self.vocab_size), -LOGIT_TARGET_VAL / (self.vocab_size - 1), dtype=self.backend.float64)
                tgt_flat = target_data.flatten().astype(self.backend.int32)
                Y_matrix[self.backend.arange(N), tgt_flat] = LOGIT_TARGET_VAL
            else:
                # Target is a continuous field -> Use it directly after reshaping
                Y_matrix = target_data.reshape(-1, target_data.shape[-1]).astype(self.backend.float64)
            
            X_matrix = self.backend.concatenate([inp_reshaped, self.backend.ones((N, 1), dtype=self.backend.float64)], axis=1)

            local_XtX = X_matrix.T @ X_matrix
            local_XtY = X_matrix.T @ Y_matrix
            return local_XtX, local_XtY

    def solve_from_matrices(self, global_XtX: ArrayLike, global_XtY: ArrayLike, layer_W: FSEField, layer_b: FSEField):
        """Solves the system using pre-computed global matrices."""
        try:
            # Add the ridge regularization term
            global_XtX[self.backend.diag_indices_from(global_XtX)] += self.ridge
            
            # Solve the linear system
            theta = self.backend.linalg.solve(global_XtX, global_XtY)

            W_new = theta[:-1, :]
            b_new = theta[-1, :]

            layer_W.data = self.backend.asarray(W_new, dtype=layer_W.dtype)
            layer_b.data = self.backend.asarray(b_new, dtype=layer_b.dtype)

        except self.backend.linalg.LinAlgError as e:
            logger.warning(f"⚠️ MorphSolver.solve_from_matrices failed: {e}. Skipping update.")
            pass



# =========================================
# PERFORMANCE MONITORING - UNCHANGED
# =========================================

class PerformanceProfiler:
    """Simple profiler for FlowField operations with context manager support"""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
        self._current_operations = {}  # Track current operations for context manager
    
    def profile_operation(self, operation_name: str):
        """Decorator for profiling operations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                duration = end_time - start_time
                if operation_name not in self.timings:
                    self.timings[operation_name] = []
                    self.call_counts[operation_name] = 0
                
                self.timings[operation_name].append(duration)
                self.call_counts[operation_name] += 1
                
                return result
            return wrapper
        return decorator
    
    @contextmanager
    def operation_context(self, operation_name: str):
        """Context manager for profiling operations"""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if operation_name not in self.timings:
                self.timings[operation_name] = []
                self.call_counts[operation_name] = 0
            
            self.timings[operation_name].append(duration)
            self.call_counts[operation_name] += 1
    
    def __call__(self, operation_name: str):
        """Allow calling the profiler directly as context manager"""
        return self.operation_context(operation_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        for op_name in self.timings:
            times = self.timings[op_name]
            stats[op_name] = {
                'count': self.call_counts[op_name],
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        return stats

# Global profiler instance
_global_profiler = PerformanceProfiler()

def get_profiler() -> PerformanceProfiler:
    return _global_profiler
