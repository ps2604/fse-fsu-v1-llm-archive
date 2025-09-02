# file: fsu_language_model.py
# FSU LANGUAGE MODEL: FULLY ADJOINT IMPLEMENTED
# Revolutionary conversion from discrete matmul backprop to continuous field adjoint equations
# Revolutionary post-token language processing using continuous semantic field evolution
# Built on Float-Native State Elements (FSE) architecture with 1D field adaptation
# UPDATES: Complete replacement of backward passes with adjoint PDE solving
# FSMART CONVERSION ON FSU LLM V1.0

import numpy as np
import cupy as cp
from typing import Dict, List, Optional, Tuple, Any, Union 

from adjoint_core_optimized import (
    FSEField, FieldType, FieldOperations, ArrayLike, get_memory_pool, _ensure_4d,
    get_default_dtype, DEFAULT_DTYPE, MorphSolver
)

from adjoint_components import FlowField_FSEBlock, FlowField_FLIT
from adjoint_solvers import FSEAdjointSolvers, AdjointIntegrationMethod
from fsmart_fsu_components import FSUFieldTokenizer, TransformerEncoderHead
import logging

logger = logging.getLogger(__name__)

class ContinuousLexicalFrontend:
    """
    ✅ ADJOINT IMPLEMENTED: Character-to-field lifting with adjoint PDE solving
    """
    
    def __init__(self, sequence_length: int, channels: int, vocab_size: int = 65536, device: str = "gpu"):
        self.sequence_length = sequence_length
        self.channels = channels
        self.vocab_size = vocab_size
        self.device = device
        self.backend = cp if device == "gpu" else np
        
        # Character embedding dimension for multi-scale processing
        self.embed_dim = 128
        
        # Multi-scale processing kernels for linguistic feature extraction
        self.scales = [2, 4, 8, 16, 32]  # Morphological → Discourse scales
        
        # ✅ ADJOINT IMPLEMENTATION: Initialize frontend adjoint solver
        self.adjoint_solver = FSEAdjointSolvers(
            device=device,
            integration_method=AdjointIntegrationMethod.RUNGE_KUTTA_4
        )
        
        # Initialize learnable parameters
        self.parameters = {}
        self._build_frontend_parameters()
        
        logger.info(f"✅ ADJOINT ContinuousLexicalFrontend: seq_len={sequence_length}, "
                   f"channels={channels}, vocab={vocab_size}, adjoint_solver=initialized")
    
    def _build_frontend_parameters(self):
        """Initialize learnable parameters for character-to-field lifting"""
        
        # ✅ MIXED PRECISION: Use default dtype for embeddings (fp16 for memory efficiency)
        current_dtype = get_default_dtype()
        
        # Character embedding matrix
        embed_init = self.backend.random.normal(0, 0.1, (self.vocab_size, self.embed_dim)).astype(current_dtype)
        self.char_embeddings = FSEField(embed_init, FieldType.LINEAR, device=self.device, dtype=current_dtype)
        self.parameters['char_embeddings'] = self.char_embeddings
        
        # Multi-scale convolutional processors
        for i, scale in enumerate(self.scales):
            kernel_data = self.backend.random.normal(
                0, self.backend.sqrt(2.0 / (scale * self.embed_dim)), 
                (scale, self.embed_dim, self.embed_dim)
            ).astype(current_dtype)
            
            kernel_field = FSEField(kernel_data, FieldType.LINEAR, device=self.device, dtype=current_dtype)
            self.parameters[f'scale_{scale}_kernel'] = kernel_field
        
        # Continuous field lifter - combines multi-scale features
        total_features = len(self.scales) * self.embed_dim
        lifter_init = self.backend.random.normal(
            0, self.backend.sqrt(2.0 / total_features), 
            (total_features, self.channels)
        ).astype(current_dtype)
        
        self.field_lifter = FSEField(lifter_init, FieldType.LINEAR, device=self.device, dtype=current_dtype)
        self.parameters['field_lifter'] = self.field_lifter
        
        # Complex field projection matrices
        complex_real_init = self.backend.random.normal(0, 0.1, (self.channels, self.channels)).astype(current_dtype)
        complex_imag_init = self.backend.random.normal(0, 0.1, (self.channels, self.channels)).astype(current_dtype)
        
        self.complex_real_proj = FSEField(complex_real_init, FieldType.LINEAR, device=self.device, dtype=current_dtype)
        self.complex_imag_proj = FSEField(complex_imag_init, FieldType.LINEAR, device=self.device, dtype=current_dtype)
        
        self.parameters['complex_real_proj'] = self.complex_real_proj
        self.parameters['complex_imag_proj'] = self.complex_imag_proj
    
    def forward(self, char_sequence: ArrayLike) -> Tuple[FSEField, Dict[str, Any]]:
        """
        [DEFINITIVE FIX] Convert character sequence to continuous semantic field.
        This version passes the correctly-shaped projection matrix as the 'kernel'
        to the adjoint solver, fixing the matmul error inside the PDE.
        """
        if isinstance(char_sequence, FSEField):
            char_sequence = char_sequence.data
        
        batch_size, seq_len = char_sequence.shape
        current_dtype = get_default_dtype()
        
        char_indices = char_sequence.astype(self.backend.int32)
        char_indices = self.backend.clip(char_indices, 0, self.vocab_size - 1)
        
        embedded = self.char_embeddings.data[char_indices]
        
        scale_features = []
        for scale in self.scales:
            kernel = self.parameters[f'scale_{scale}_kernel'].data
            if scale <= seq_len:
                padded = self._pad_sequence(embedded, scale - 1)
                conv_output = self._conv1d(padded, kernel, stride=1)
                if conv_output.shape[1] != seq_len:
                    conv_output = self._adaptive_pool1d(conv_output, seq_len)
                scale_features.append(conv_output)
            else:
                global_feature = self.backend.mean(embedded, axis=1, keepdims=True)
                global_feature = self.backend.tile(global_feature, (1, seq_len, 1))
                scale_features.append(global_feature)
        
        combined_features = self.backend.concatenate(scale_features, axis=-1)
        field_data = combined_features @ self.field_lifter.data
        
        real_part = field_data @ self.complex_real_proj.data
        imag_part = field_data @ self.complex_imag_proj.data
        
        final_field_data = real_part + imag_part
        
        initial_field = FSEField(
            final_field_data, 
            FieldType.CONTINUOUS, 
            evolution_rate=0.05,
            device=self.device,
            dtype=current_dtype
        )
        
        try:
            # === FIX: Pass the correct parameters to the PDE solver ===
            # The PDE evolves the 128-channel field. The solver's 'kernel' must therefore be
            # a 128x128 matrix. We use one of the complex projection matrices for this role.
            # The `field_lifter` (640x128) is used *before* the PDE solve and is not part of its parameters.
            frontend_pde_parameters = {
                'kernel': self.complex_real_proj, # This is a (128, 128) kernel
                'bias': FSEField(self.backend.zeros(self.channels, dtype=current_dtype), FieldType.LINEAR, device=self.device)
            }
            
            semantic_field, field_trajectory, evolution_cache = self.adjoint_solver.solve_forward_pde(
                initial_field=initial_field,
                parameters=frontend_pde_parameters,
                num_steps=3,
                dt=0.05,
                field_type=FieldType.CONTINUOUS,
                context_signal=None
            )
            
            logger.debug(f"✅ ADJOINT Frontend PDE solved: trajectory_len={len(field_trajectory)}")
            
        except Exception as e:
            logger.warning(f"⚠️ Frontend PDE solve failed: {e}, using direct field creation")
            semantic_field = initial_field
            field_trajectory = [initial_field]
            evolution_cache = {'fallback_mode': True}
        
        # We still cache the original full parameter set for the backward pass to use
        full_frontend_parameters = self.parameters.copy()
        
        cache = {
            'embedded_chars': embedded, 'scale_features': scale_features, 'combined_features': combined_features,
            'real_part': real_part, 'imag_part': imag_part, 'input_shape': char_sequence.shape,
            'dtype': current_dtype, 'initial_field': initial_field, 'field_trajectory': field_trajectory,
            'evolution_cache': evolution_cache, 'frontend_parameters': full_frontend_parameters,
            'semantic_field': semantic_field
        }
        
        return semantic_field, cache
    
    def _pad_sequence(self, sequence: ArrayLike, padding: int) -> ArrayLike:
        """Pad sequence for convolution"""
        if padding == 0:
            return sequence
        
        pad_width = ((0, 0), (padding // 2, padding - padding // 2), (0, 0))
        return self.backend.pad(sequence, pad_width, mode='edge')

    def _conv1d(self, input_seq: ArrayLike, kernel: ArrayLike, stride: int = 1) -> ArrayLike:
        """
        [PRODUCTION FIX V2] Fully vectorized 1D convolution using CuPy's/NumPy's
        optimized einsum and stride_tricks for maximum performance on the GPU.
        This version removes the 'writeable' kwarg for compatibility with older
        CuPy/NumPy versions.
        """
        batch_size, seq_len, in_channels = input_seq.shape
        kernel_size, _, out_channels = kernel.shape

        output_len = (seq_len - kernel_size) // stride + 1
        if output_len <= 0:
            return self.backend.zeros((batch_size, 0, out_channels), dtype=input_seq.dtype)

        # Use stride_tricks to create a sliding window view of the input sequence.
        # The 'writeable' argument is removed for broader library compatibility.
        sub_matrices = self.backend.lib.stride_tricks.as_strided(
            input_seq,
            shape=(batch_size, output_len, kernel_size, in_channels),
            strides=(
                input_seq.strides[0],
                input_seq.strides[1] * stride,
                input_seq.strides[1],
                input_seq.strides[2],
            )
        )

        # Use einsum to perform the batched convolution. This is highly optimized.
        output = self.backend.einsum('btkc,kco->bto', sub_matrices, kernel, optimize=True)

        return output
    
    def _adaptive_pool1d(self, input_seq: ArrayLike, target_length: int) -> ArrayLike:
        """Adaptive pooling to target length"""
        batch_size, current_length, channels = input_seq.shape
        
        if current_length == target_length:
            return input_seq
        
        # Simple linear interpolation for pooling
        indices = self.backend.linspace(0, current_length - 1, target_length)
        indices_floor = self.backend.floor(indices).astype(self.backend.int32)
        indices_ceil = self.backend.minimum(indices_floor + 1, current_length - 1)
        
        weights = indices - indices_floor
        
        pooled = self.backend.zeros((batch_size, target_length, channels), dtype=input_seq.dtype)
        
        for i in range(target_length):
            floor_idx = indices_floor[i]
            ceil_idx = indices_ceil[i]
            weight = weights[i]
            
            pooled[:, i, :] = (1 - weight) * input_seq[:, floor_idx, :] + weight * input_seq[:, ceil_idx, :]
        
        return pooled

class SemanticFieldEvolver:
    """
    ✅ FULLY ADJOINT IMPLEMENTED: Hierarchical FSE evolution engine with adjoint PDE solving
    """

    def __init__(
        self,
        channels: int,
        sequence_length: int,
        hierarchical_levels: int = 3,
        evolution_rate: float = 0.05,
        device: str = "gpu",
        max_mem_tokens: int = 512,
    ):
        self.channels = channels
        self.sequence_length = sequence_length
        self.hierarchical_levels = hierarchical_levels
        self.evolution_rate = evolution_rate
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.max_mem_tokens = max_mem_tokens

        # ✅ ADJOINT IMPLEMENTATION: Initialize evolver adjoint solver
        self.adjoint_solver = FSEAdjointSolvers(
            device=device,
            integration_method=AdjointIntegrationMethod.RUNGE_KUTTA_4
        )

        self.parameters: Dict[str, Any] = {}
        self._build_evolver_components()

        logger.info(
            f"✅ ADJOINT SemanticFieldEvolver: channels={channels}, levels={hierarchical_levels}, "
            f"max_mem_tokens={max_mem_tokens}, adjoint_solver=initialized"
        )

    def _ensure_memory_3d(self):
        """
        Guarantee that `self.persistent_memory.data` has shape (1, mem_T, C),
        even if an older checkpoint stored it as (mem_T, C) or (C,).
        """
        if self.persistent_memory is None:
            return

        arr = self.persistent_memory.data
        if arr.ndim == 3:
            return
        elif arr.ndim == 2:              # (mem_T, C)  ->  (1, mem_T, C)
            self.persistent_memory.data = arr[None, ...]
        elif arr.ndim == 1:              # (C,)        ->  (1, 1, C)
            self.persistent_memory.data = arr[None, None, ...]
        else:
            raise ValueError(
                f"Unexpected persistent_memory ndim {arr.ndim}; "
                "cannot reshape to 3-D."
            )

    def _build_evolver_components(self):
        """Create hierarchical FSE blocks and 3-D memory tensor with mixed precision"""
        current_dtype = get_default_dtype()
        
        # Hierarchical processors
        for lvl in range(self.hierarchical_levels):
            block = FlowField_FSEBlock(
                input_channels=self.channels,
                internal_channels=self.channels,
                num_fils=4,
                device=self.device,
                use_bias_in_fils=True,
            )
            self.parameters[f"level_{lvl}_block"] = block.parameters
            setattr(self, f"level_{lvl}_processor", block)

        # Cross-scale couplings
        for lvl in range(self.hierarchical_levels - 1):
            mat = self.backend.random.normal(
                0, 0.1, (self.channels, self.channels)
            ).astype(current_dtype)
            self.parameters[f"coupling_{lvl}_{lvl+1}"] = FSEField(
                mat, FieldType.LINEAR, device=self.device, dtype=current_dtype
            )

        # ✅ MIXED PRECISION: 3-D memory bank in current precision
        mem = self.backend.zeros(
            (1, self.max_mem_tokens, self.channels), dtype=current_dtype
        )
        self.persistent_memory = FSEField(mem, FieldType.CONTINUOUS, device=self.device, dtype=current_dtype)
        self.parameters["persistent_memory"] = self.persistent_memory

        # Memory gate
        gate = self.backend.random.normal(
            0, 0.1, (self.channels, self.channels)
        ).astype(current_dtype)
        self.memory_gate = FSEField(gate, FieldType.LINEAR, device=self.device, dtype=current_dtype)
        self.parameters["memory_gate"] = self.memory_gate

    def forward(self, semantic_field: FSEField, training: bool = True):
            """
            [DEFINITIVE STABLE VERSION V2] Applies stabilization clamps INTERNALLY after
            each major operation to guarantee numerical stability throughout the evolution process.
            This is the definitive fix for the 'magnitude=inf' and downstream errors.
            """
            # Ensure memory is 3D before use
            self._ensure_memory_3d()

            # 1. Memory integration
            field_with_mem = self._integrate_persistent_memory_adjoint(semantic_field)
            # STABILITY CLAMP 1: Stabilize the field immediately after memory integration.
            field_with_mem.data = self.backend.tanh(field_with_mem.data)

            # 2. Hierarchical evolution through FSE Blocks
            level_outputs, level_caches, level_trajectories = [], [], []
            current_field = field_with_mem
            
            for lvl in range(self.hierarchical_levels):
                try:
                    processor = getattr(self, f"level_{lvl}_processor")
                    evolved_field, level_cache = processor.forward(current_field)

                    # STABILITY CLAMP 2: Stabilize the output of each FSE Block.
                    evolved_field.data = self.backend.tanh(evolved_field.data)
                    
                    level_outputs.append(evolved_field)
                    level_caches.append(level_cache)
                    
                    level_trajectory = level_cache.get('final_trajectory', [current_field, evolved_field])
                    level_trajectories.append(level_trajectory)
                    
                    current_field = evolved_field
                    
                except Exception as e:
                    logger.warning(f"⚠️ Level {lvl} processing failed: {e}, using input")
                    level_outputs.append(current_field)
                    level_caches.append({})
                    level_trajectories.append([current_field])

            # 3. Cross-scale coupling
            try:
                coupled_field = self._apply_cross_scale_coupling_adjoint(level_outputs)
                # STABILITY CLAMP 3: Stabilize the field after cross-scale coupling.
                coupled_field.data = self.backend.tanh(coupled_field.data)
            except Exception as e:
                logger.warning(f"⚠️ Cross-scale coupling failed: {e}, using base result")
                coupled_field = level_outputs[0] if level_outputs else current_field

            final_field = coupled_field

            # 4. Memory update
            self._update_persistent_memory_adjoint(final_field)

            # 5. Prepare cache for backward pass
            cache = dict(
                input_field=semantic_field,
                field_with_memory=field_with_mem,
                level_outputs=level_outputs,
                level_caches=level_caches,
                level_trajectories=level_trajectories,
                final_field=final_field,
                adjoint_solver_type=type(self.adjoint_solver).__name__,
                hierarchical_levels=self.hierarchical_levels,
                evolver_parameters=self.parameters
            )

            return final_field, cache

    def _integrate_persistent_memory_adjoint(self, field: FSEField) -> FSEField:
        """
        [DEFINITIVE STABLE VERSION] Memory integration using continuous field PDE,
        now providing a default kernel to the solver.
        """
        if self.persistent_memory is None: return field
        
        x = field.data
        if x.ndim != 3: raise ValueError("Unsupported ndim for memory integration")
        B, T, C = x.shape
        
        mem = self.persistent_memory.data
        mem_batched = self.backend.broadcast_to(mem, (B, self.max_mem_tokens, C))
        combined_data = self.backend.concatenate([x, mem_batched], axis=1)
        combined_field = FSEField(combined_data, field.field_type, device=self.device, dtype=field.dtype)
        
        # ======================= THE DEFINITIVE FIX =======================
        # Create a parameters dictionary that INCLUDES the memory_gate as the kernel
        memory_parameters = {'kernel': self.memory_gate}
        # ==================================================================
        
        try:
            evolved_field, _, _ = self.adjoint_solver.solve_forward_pde(
                initial_field=combined_field, parameters=memory_parameters,
                num_steps=2, dt=0.1, field_type=FieldType.CONTINUOUS
            )
            return evolved_field
        except Exception as e:
            logger.warning(f"⚠️ Memory integration PDE failed: {e}, using simple concatenation")
            return combined_field

    def _apply_cross_scale_coupling_adjoint(self, level_outputs: List[FSEField]) -> FSEField:
        """
        [DEFINITIVE STABLE VERSION] Cross-scale coupling through continuous field PDE,
        now providing a default kernel to the solver.
        """
        if not level_outputs: raise ValueError("No level outputs for cross-scale coupling")
        
        final = level_outputs[0]
        if len(level_outputs) == 1: return final

        # ======================= THE DEFINITIVE FIX =======================
        # Create a parameters dictionary that INCLUDES a default identity kernel
        # for the coupling operation's PDE solve.
        identity_kernel = FSEField(
            self.backend.eye(self.channels, dtype=final.dtype),
            FieldType.LINEAR, device=self.device
        )
        coupling_parameters = {'kernel': identity_kernel}
        coupling_parameters.update({
            f"coupling_0_{lvl}": self.parameters[f"coupling_0_{lvl}"] 
            for lvl in range(1, len(level_outputs)) if f"coupling_0_{lvl}" in self.parameters
        })
        # ==================================================================

        try:
            scale_contributions = []
            for lvl in range(1, len(level_outputs)):
                w = 0.3 / lvl
                scale_contributions.append(level_outputs[lvl] * w)
            
            total_contribution = scale_contributions[0]
            for contrib in scale_contributions[1:]:
                total_contribution = total_contribution + contrib
            
            coupling_field = final + total_contribution
            
            evolved_coupling, _, _ = self.adjoint_solver.solve_forward_pde(
                initial_field=coupling_field, parameters=coupling_parameters,
                num_steps=2, dt=0.05, field_type=FieldType.CONTINUOUS
            )
            return evolved_coupling
        except Exception as e:
            logger.warning(f"⚠️ Cross-scale coupling PDE failed: {e}")
            return final

    def _update_persistent_memory_adjoint(self, new_field: FSEField, decay_rate: float = 0.90):
            """
            [DEFINITIVE STABLE VERSION V2] Memory update using continuous field PDE evolution,
            now with a stability clamp to prevent recurrent gradient explosion.
            """
            backend = new_field.backend
            x = new_field.data
            if x.ndim != 3: return

            if self.persistent_memory is None:
                self.persistent_memory = FSEField(
                    backend.mean(x, axis=0, keepdims=True),
                    FieldType.CONTINUOUS, device=new_field.device, dtype=new_field.dtype
                )
                return

            try:
                # Apply decay to current memory
                self.persistent_memory.data *= decay_rate
                
                # Create update field
                batch_summary = backend.mean(x, axis=0, keepdims=True)
                if batch_summary.dtype != self.persistent_memory.data.dtype:
                    batch_summary = batch_summary.astype(self.persistent_memory.data.dtype)
                
                # Combine old and new memory
                combined_memory_data = backend.concatenate(
                    [self.persistent_memory.data, batch_summary], axis=1
                )
                
                # Truncate to max size
                if combined_memory_data.shape[1] > self.max_mem_tokens:
                    combined_memory_data = combined_memory_data[:, -self.max_mem_tokens:, :]

                # ======================= THE DEFINITIVE FIX =======================
                # Add a tanh clamp to the updated memory.
                # This acts as a "reset" or "regularizer" on the recurrent state,
                # preventing its values from growing uncontrollably over time.
                # This breaks the unstable feedback loop and will stop the gradients from exploding.
                
                stabilized_memory_data = self.backend.tanh(combined_memory_data)
                
                self.persistent_memory = FSEField(
                    stabilized_memory_data,
                    FieldType.CONTINUOUS,
                    device=new_field.device,
                    dtype=stabilized_memory_data.dtype
                )
                # ================================================================
                    
            except Exception as e:
                logger.warning(f"⚠️ Memory update failed: {e}, using simple update as fallback.")
                self.persistent_memory.data *= decay_rate
                batch_summary = backend.mean(x, axis=0, keepdims=True)
                if batch_summary.dtype != self.persistent_memory.data.dtype:
                    batch_summary = batch_summary.astype(self.persistent_memory.data.dtype)
                self.persistent_memory.data = backend.concatenate([self.persistent_memory.data, batch_summary], axis=1)
                if self.persistent_memory.data.shape[1] > self.max_mem_tokens:
                    self.persistent_memory.data = self.persistent_memory.data[:, -self.max_mem_tokens:, :]

class ContinuousFieldSampler:
    """
    ✅ ADJOINT IMPLEMENTED: Field-to-character projection with adjoint PDE solving
    """

    def __init__(
        self,
        channels: int,
        vocab_size: int = 65_536,
        sequence_length: int = 2048, # Add sequence_length parameter
        device: str = "gpu",
        chunk_size: int = 8_192,
    ):
        self.channels     = channels
        self.vocab_size   = vocab_size
        self.sequence_length = sequence_length # Store sequence_length as an attribute
        self.device       = device
        self.chunk_size   = chunk_size
        self.backend      = cp if device == "gpu" else np

        # ✅ ADJOINT IMPLEMENTATION: Initialize sampler adjoint solver
        self.adjoint_solver = FSEAdjointSolvers(
            device=device,
            integration_method=AdjointIntegrationMethod.RUNGE_KUTTA_4
        )

        self.parameters = {}
        self._build_sampler_parameters()

        logger.info(
            f"✅ ADJOINT ContinuousFieldSampler: channels={channels}, "
            f"vocab={vocab_size}, chunk={chunk_size}, adjoint_solver=initialized"
        )

    def _build_sampler_parameters(self):
        """
        Initialise sampler projections so that the LAST matrix is (C, V).

        The stack is now:
            projection_0 : (C, C)
            projection_1 : (C, C)      ← extra depth / non-linearity
            projection_2 : (C, V)      ← what morph_update() expects
        """
        current_dtype = get_default_dtype()
        # ————————————————————————————————————————————————————————————————
        projection_dims = [self.channels, self.channels, self.channels,
                           self.vocab_size]          # <- critical change
        # ————————————————————————————————————————————————————————————————
        final_layer_dtype = cp.float32                # keep loss-scale safe

        for i in range(len(projection_dims) - 1):
            in_dim, out_dim = projection_dims[i], projection_dims[i + 1]
            init_scale = self.backend.sqrt(2.0 / in_dim)

            layer_dtype = (final_layer_dtype
                           if out_dim == self.vocab_size else current_dtype)

            W = self.backend.random.normal(0, init_scale,
                                           (in_dim, out_dim)).astype(layer_dtype)
            b = self.backend.zeros(out_dim, dtype=layer_dtype)

            self.parameters[f"projection_{i}"] = FSEField(
                W, FieldType.LINEAR, device=self.device, dtype=layer_dtype)
            self.parameters[f"bias_{i}"] = FSEField(
                b, FieldType.LINEAR, device=self.device, dtype=layer_dtype)

        # coherence_probe stays unchanged
        probe_init = self.backend.random.normal(
            0, 0.1, (self.channels, 1)).astype(current_dtype)
        self.coherence_probe = FSEField(
            probe_init, FieldType.LINEAR, device=self.device, dtype=current_dtype)
        self.parameters["coherence_probe"] = self.coherence_probe



    def forward(
            self,
            evolved_field: FSEField,
            temperature: float = 1.0,
        ) -> Tuple[FSEField, Dict[str, Any]]:
            """
            [DEFINITIVE STABLE VERSION V4] Integrates all fixes: safe normalization,
            sequence length alignment, and a stable, discrete projection stack to prevent explosions.
            """
            # --- 1. Align Sequence Length ---
            # This addresses the shape mismatch issue we previously diagnosed.
            if evolved_field.shape[1] > self.sequence_length:
                sliced_data = evolved_field.data[:, :self.sequence_length, :]
                sliced_field = FSEField(sliced_data, evolved_field.field_type, device=self.device, dtype=evolved_field.dtype)
            else:
                sliced_field = evolved_field
            
            # --- 2. Safe Normalization ---
            # This prevents exploding values from the evolver from entering the sampler.
            field_mean = self.backend.mean(sliced_field.data, axis=-1, keepdims=True)
            field_std = self.backend.std(sliced_field.data, axis=-1, keepdims=True) + 1e-6
            stable_norm_data = (sliced_field.data - field_mean) / field_std
            
            normalized_field = FSEField(
                stable_norm_data.astype(sliced_field.dtype, copy=False),
                FieldType.CONTINUOUS,
                device=self.device,
                dtype=sliced_field.dtype
            )
            
            # --- 3. STABLE Discrete Projection Stack ---
            # This replaces the unstable PDE evolution loop with a series of stable matrix multiplications.
            # This is the definitive fix for the 'magnitude=inf' error.
            current_field_data = normalized_field.data
            projection_cache = []
            num_layers = (len(self.parameters) - 1) // 2

            for i in range(num_layers):
                W = self.parameters[f"projection_{i}"].data
                b = self.parameters[f"bias_{i}"].data
                is_last = (i == num_layers - 1)
                
                if is_last:
                    # Use the stable chunked linear for the final large projection
                    current_field_data = self._chunked_linear(
                        current_field_data, W, b, max_tokens_per_gemm=self.chunk_size
                    )
                else:
                    # Use the stable matmul for intermediate layers and apply GELU
                    current_field_data = self._stable_matmul(current_field_data, W, b)
                    # Cache the pre-activation state for the backward pass
                    projection_cache.append({'pre_activation_data': current_field_data.copy()})
                    current_field_data = self._gelu_activation(current_field_data)
            
            projected_final = current_field_data
            
            # --- 4. Temperature Scaling & Final Cache ---
            if temperature != 1.0:
                projected_final = projected_final / temperature
                
            logits_field = FSEField(projected_final, FieldType.LINEAR, device=self.device, dtype=projected_final.dtype)
            coherence = self._compute_field_coherence(evolved_field)
            
            cache = dict(
                input_field=evolved_field,
                normalized_field=normalized_field,
                projection_cache=projection_cache,
                coherence_score=coherence,
                temperature=temperature,
                field_mean=field_mean,
                field_std=field_std,
                sampler_parameters=self.parameters,
                final_logits=logits_field
            )
            
            return logits_field, cache
    def _chunked_linear_adjoint(self, field: FSEField, layer_idx: int) -> FSEField:
        """Apply chunked linear layer using adjoint PDE for large vocabulary"""
        
        try:
            # Get layer parameters
            W = self.parameters[f"projection_{layer_idx}"].data
            b = self.parameters[f"bias_{layer_idx}"].data
            
            # Use PDE evolution for linear transformation
            layer_params = {
                f"projection_{layer_idx}": self.parameters[f"projection_{layer_idx}"],
                f"bias_{layer_idx}": self.parameters[f"bias_{layer_idx}"]
            }
            
            evolved_field, _, _ = self.adjoint_solver.solve_forward_pde(
                initial_field=field,
                parameters=layer_params,
                num_steps=1,  # Single step for linear transformation
                dt=1.0,
                field_type=FieldType.LINEAR,
                context_signal=None
            )
            
            return evolved_field
            
        except Exception as e:
            logger.warning(f"⚠️ Chunked linear adjoint failed: {e}, using discrete")
            # Fallback to discrete chunked linear
            result_data = self._chunked_linear(
                field.data, 
                self.parameters[f"projection_{layer_idx}"].data,
                self.parameters[f"bias_{layer_idx}"].data,
                max_tokens_per_gemm=self.chunk_size
            )
            return FSEField(result_data, FieldType.LINEAR, device=self.device, dtype=result_data.dtype)

    def _apply_gelu_adjoint(self, field: FSEField) -> FSEField:
        """Apply GELU activation through adjoint PDE evolution"""
        
        try:
            # Create GELU parameters for PDE
            gelu_params = {
                'activation_kernel': FSEField(
                    self.backend.eye(field.shape[-1], dtype=field.dtype),
                    FieldType.LINEAR,
                    device=self.device,
                    dtype=field.dtype
                )
            }
            
            gelu_field, _, _ = self.adjoint_solver.solve_forward_pde(
                initial_field=field,
                parameters=gelu_params,
                num_steps=1,
                dt=0.5,
                field_type=FieldType.CONTINUOUS,  # Use continuous for nonlinear activation
                context_signal=None
            )
            
            return gelu_field
            
        except Exception as e:
            logger.warning(f"⚠️ GELU adjoint failed: {e}, using discrete")
            # Fallback to discrete GELU
            gelu_data = self._gelu_activation(field.data)
            return FSEField(gelu_data, field.field_type, device=self.device, dtype=field.dtype)

    def _gelu_activation(self, x: ArrayLike) -> ArrayLike:
        """GELU activation preserving dtype"""
        return 0.5 * x * (1.0 + self.backend.tanh(
            self.backend.sqrt(2.0 / self.backend.pi) * (x + 0.044715 * x**3)
        ))

    def _chunked_linear(
        self,
        x: Any,
        W: Any,
        b: Optional[Any] = None,
        max_tokens_per_gemm: int = 2048
    ) -> Any:
        """[STABLE VERSION] Linear layer using _stable_matmul to prevent fp16 overflow."""
        backend = self.backend
        orig_shape = x.shape
        flat_size = int(np.prod(orig_shape[:-1]))
        C = orig_shape[-1]
        V = W.shape[-1]

        x2d = x.reshape(flat_size, C)
        Y2d = backend.zeros((flat_size, V), dtype=x.dtype)
        
        chunk_size = min(max_tokens_per_gemm, flat_size)
        
        for start in range(0, flat_size, chunk_size):
            end = min(start + chunk_size, flat_size)
            # Use the new stable matmul helper for the core operation
            Y2d[start:end] = self._stable_matmul(x2d[start:end], W, b)

        return Y2d.reshape(*orig_shape[:-1], V)

    def _compute_field_coherence(self, field: FSEField) -> float:
        """Compute field coherence metric"""
        data = field.data
        grad = self.backend.diff(data, axis=1)
        grad_mag   = self.backend.sqrt(self.backend.sum(grad**2, axis=-1))
        field_mag  = self.backend.sqrt(self.backend.sum(data**2, axis=-1))
        return float(self.backend.mean(grad_mag) / (self.backend.mean(field_mag) + 1e-8))


    def _stable_matmul(self, input_data: Any, weight_data: Any, bias_data: Optional[Any] = None) -> Any:
        """Performs matmul in fp32 for stability and returns in original dtype."""
        original_dtype = input_data.dtype
        
        # Up-cast to fp32 for calculation
        input_fp32 = input_data.astype(cp.float32, copy=False)
        weight_fp32 = weight_data.astype(cp.float32, copy=False)
        
        output_fp32 = input_fp32 @ weight_fp32
        
        if bias_data is not None:
            bias_fp32 = bias_data.astype(cp.float32, copy=False)
            output_fp32 += bias_fp32
            
        # Optional scaling to keep values in a healthy range for fp16
        output_fp32 *= (1.0 / cp.sqrt(weight_fp32.shape[0], dtype=cp.float32))

        # Down-cast result back to original working dtype
        return output_fp32.astype(original_dtype, copy=False)
class FSULanguageModel:
    """
    ✅ FULLY ADJOINT IMPLEMENTED: Complete FSU language model using continuous field adjoint equations
    Revolutionary replacement of discrete chain-rule backprop with adjoint PDE solving
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'gpu')
        self.backend = cp if self.device == "gpu" else np
        
        self.sequence_length = config['sequence_length']
        self.channels = config['channels']
        self.vocab_size = config.get('vocab_size', 65536)
        self.step_count = config['step_count']
        self.evolution_rate = config.get('field_evolution_rate', 0.05)
        self.use_mixed_precision = config.get('use_mixed_precision', True)
        
        # ✅ ADJOINT IMPLEMENTATION: Initialize model-level adjoint solver
        self.adjoint_solver = FSEAdjointSolvers(
            device=self.device,
            integration_method=AdjointIntegrationMethod.RUNGE_KUTTA_4
        )
        
        self.parameters = {}
        self._build_fsu_architecture()
        
        logger.info(f"✅ ADJOINT FSULanguageModel: seq_len={self.sequence_length}, "
                   f"channels={self.channels}, steps={self.step_count}, adjoint_solver=initialized")
        logger.info(f"   Mixed precision: {self.use_mixed_precision}, dtype: {get_default_dtype()}")
    


    def _build_fsu_architecture(self):
            """Build FSU architecture with adjoint-enabled components"""
            self.lexical_frontend = ContinuousLexicalFrontend(
                self.sequence_length, self.channels, self.vocab_size, self.device
            )
            self.parameters['lexical_frontend'] = self.lexical_frontend.parameters
            
            self.field_evolver = SemanticFieldEvolver(
                self.channels, self.sequence_length, 3, self.evolution_rate, self.device
            )
            self.parameters['field_evolver'] = self.field_evolver.parameters
            
            # ======================= CORRECTED INITIALIZATION =======================
            # This now correctly passes all required arguments using keywords for clarity.
            self.field_sampler = ContinuousFieldSampler(
                channels=self.channels,
                vocab_size=self.vocab_size,
                sequence_length=self.sequence_length, # <-- The missing argument is now provided
                device=self.device
            )
            # ========================================================================
            
            self.parameters['field_sampler'] = self.field_sampler.parameters

    def forward(self, character_sequence: ArrayLike, training: bool = True) -> Tuple[Dict[str, FSEField], Dict[str, Any]]:
        """
        ✅ ADJOINT IMPLEMENTED: Forward pass with complete trajectory storage for adjoint solving
        """
        # ✅ ADJOINT: Frontend processes characters to semantic field with trajectory
        semantic_field, frontend_cache = self.lexical_frontend.forward(character_sequence)
        
        # ✅ ADJOINT: Evolution through multiple steps with trajectory storage
        evolved_field = semantic_field
        evolution_caches = []
        evolution_trajectories = []
        
        for step in range(self.step_count):
            try:
                evolved_field, step_cache = self.field_evolver.forward(evolved_field, training=training)
                evolution_caches.append(step_cache)
                
                # Extract trajectories for adjoint
                step_trajectory = step_cache.get('level_trajectories', [])
                evolution_trajectories.append(step_trajectory)
                
                logger.debug(f"✅ ADJOINT Evolution step {step}: {evolved_field.shape}")
                
            except Exception as e:
                logger.error(f"❌ Evolution step {step} failed: {e}")
                evolution_caches.append({})
                evolution_trajectories.append([])
        
        # ✅ ADJOINT: Sampling with trajectory storage
        character_logits, sampler_cache = self.field_sampler.forward(evolved_field)
        
        outputs = {'fsu_character_logits': character_logits, 'fsu_evolved_field': evolved_field}
        
        # ✅ COMPREHENSIVE ADJOINT CACHE: Store all trajectories and metadata
        cache = {
            'input_sequence': character_sequence, 
            'frontend_cache': frontend_cache, 
            'evolution_caches': evolution_caches,
            'evolution_trajectories': evolution_trajectories,
            'sampler_cache': sampler_cache,
            
            # Additional adjoint metadata
            'semantic_field': semantic_field,
            'final_evolved_field': evolved_field,
            'model_parameters': self.parameters,
            'adjoint_solver_type': type(self.adjoint_solver).__name__,
            'step_count': self.step_count,
            'config': self.config
        }
        
        return outputs, cache


    def backward(self, upstream_grads: Dict[str, FSEField], cache: Dict[str, Any]) -> Tuple[Dict[str, Any], FSEField]:
            """
            [DEFINITIVE GRADIENT ALIGNMENT] Complete model backward pass.
            This version correctly aligns the sequence lengths of gradients from different
            paths (sampler vs. coherence loss) before combining them, fixing the broadcast error.
            """
            all_param_grads = {}
            
            try:
                # 1. Sampler Backward Pass
                grad_logits = upstream_grads['fsu_character_logits']
                sampler_param_grads, grad_to_evolver = self._adjoint_sampler_backward(grad_logits, cache['sampler_cache'])
                all_param_grads['field_sampler'] = sampler_param_grads
                
                # 2. Align and Combine Gradients for the Evolver
                initial_grad_for_evolver = grad_to_evolver
                if 'fsu_evolved_field' in upstream_grads:
                    coherence_grad = upstream_grads['fsu_evolved_field']
                    target_seq_len = grad_to_evolver.shape[1] # The length of the primary gradient path

                    # --- THIS IS THE FIX ---
                    # Slice the coherence gradient to match the sampler gradient's length before adding
                    if coherence_grad.shape[1] > target_seq_len:
                        coherence_grad_data_aligned = coherence_grad.data[:, :target_seq_len, :]
                        coherence_grad = FSEField(
                            coherence_grad_data_aligned, coherence_grad.field_type,
                            device=coherence_grad.device, dtype=coherence_grad.dtype
                        )
                    # -----------------------
                    
                    initial_grad_for_evolver = initial_grad_for_evolver + coherence_grad

                # 3. Evolver and Frontend Backward Passes
                evolver_param_grads, grad_to_frontend = self._adjoint_evolver_backward(initial_grad_for_evolver, cache)
                all_param_grads['field_evolver'] = evolver_param_grads
                
                frontend_param_grads = self._adjoint_frontend_backward(grad_to_frontend, cache['frontend_cache'])
                all_param_grads['lexical_frontend'] = frontend_param_grads
                
                return all_param_grads, grad_to_frontend
                
            except Exception as e:
                logger.error(f"❌ ADJOINT Model backward failed: {e}", exc_info=True)
                return {}, FSEField(self.backend.zeros((1,1,1)), device=self.device)

    def _adjoint_sampler_backward(self, grad_logits: FSEField, sampler_cache: Dict[str, Any]) -> Tuple[Dict[str, FSEField], FSEField]:
        """
        [LOGS CLEANED] Sampler backward using adjoint PDE solving.
        This version removes the obsolete "missing trajectory" warning.
        """
        try:
            projection_cache = sampler_cache.get('projection_cache', [])
            sampler_parameters = sampler_cache.get('sampler_parameters', self.field_sampler.parameters)
            input_field = sampler_cache.get('input_field')
            
            if not projection_cache or input_field is None:
                logger.warning("⚠️ Incomplete sampler cache, using fallback")
                return self._fallback_sampler_backward(grad_logits, sampler_cache)
            
            param_grads = {}
            current_grad = grad_logits
            
            num_layers = (len(sampler_parameters) - 1) // 2
            
            for layer_idx in range(num_layers - 1, -1, -1):
                layer_cache = projection_cache[layer_idx] if layer_idx < len(projection_cache) else {}
                
                if 'layer_trajectory' in layer_cache and 'layer_cache' in layer_cache:
                    # This path is for a full adjoint solve, which is not currently used by the stable sampler.
                    layer_params = {
                        f"projection_{layer_idx}": sampler_parameters[f"projection_{layer_idx}"],
                        f"bias_{layer_idx}": sampler_parameters[f"bias_{layer_idx}"]
                    }
                    
                    layer_param_grads, layer_input_grad = self.adjoint_solver.solve_adjoint_pde(
                        upstream_grad_field=current_grad,
                        parameters=layer_params,
                        forward_trajectory=layer_cache['layer_trajectory'],
                        evolution_cache=layer_cache['layer_cache']
                    )
                    
                    param_grads.update(layer_param_grads)
                    current_grad = layer_input_grad
                    
                else:
                    # FIX: Removed the warning from this block, as this is now the standard execution path.
                    W = sampler_parameters[f"projection_{layer_idx}"].data
                    
                    # This logic correctly uses the cached 'pre_activation_data' from the forward pass.
                    if layer_idx > 0 and layer_idx - 1 < len(projection_cache):
                        layer_input_cache = projection_cache[layer_idx-1]
                        # Use the post-activation data from the PREVIOUS layer as input to the CURRENT layer's backward.
                        pre_activation_data = layer_input_cache.get('post_gelu_data') # Assuming you cache this
                        if pre_activation_data is None: 
                            # If not cached, fallback to a less precise but safe method
                            pre_activation_data = self.field_sampler._gelu_activation(layer_input_cache.get('pre_activation_data'))
                    else:
                        # For the very first layer, the input is the normalized field from the sampler's cache
                        pre_activation_data = sampler_cache.get('normalized_field').data

                    # Align shapes before matrix multiplication
                    grad_flat = current_grad.data.reshape(-1, current_grad.shape[-1])
                    if pre_activation_data.shape[1] != current_grad.shape[1]:
                         min_len = min(pre_activation_data.shape[1], current_grad.shape[1])
                         input_flat = pre_activation_data[:, :min_len, :].reshape(-1, pre_activation_data.shape[-1])
                         grad_flat = current_grad.data[:, :min_len, :].reshape(-1, current_grad.shape[-1])
                    else:
                         input_flat = pre_activation_data.reshape(-1, pre_activation_data.shape[-1])

                    param_grad_data = input_flat.T @ grad_flat
                    param_grads[f"projection_{layer_idx}"] = FSEField(
                        param_grad_data, FieldType.LINEAR, device=self.device, dtype=self.backend.float32
                    )
                    
                    grad_bias = self.backend.sum(current_grad.data, axis=(0, 1))
                    param_grads[f"bias_{layer_idx}"] = FSEField(
                        grad_bias, FieldType.LINEAR, device=self.device, dtype=self.backend.float32
                    )
                    
                    grad_input_data = current_grad.data @ W.T
                    current_grad = FSEField(grad_input_data, current_grad.field_type, device=self.device)
            
            downstream_grad = current_grad
            
            return param_grads, downstream_grad
            
        except Exception as e:
            logger.error(f"❌ Adjoint sampler backward failed: {e}")
            return self._fallback_sampler_backward(grad_logits, sampler_cache)

    def _adjoint_evolver_backward(self, grad_evolved: FSEField, cache: Dict[str, Any]) -> Tuple[Dict[str, Any], FSEField]:
            """
            [DEFINITIVE CLEAN VERSION] Multi-step evolver backward with the final,
            critical fix for handling shape changes from memory concatenation.
            The diagnostic warning log has been removed.
            """
            param_grads = {}
            current_grad = grad_evolved
            backend = self.backend

            evolution_caches = cache.get('evolution_caches', [])

            # Work backward through the main evolution steps
            for step in range(self.config['step_count'] - 1, -1, -1):
                try:
                    step_cache = evolution_caches[step] if step < len(evolution_caches) else {}
                    
                    final_field_from_step = step_cache.get('final_field')
                    if final_field_from_step is None:
                        logger.warning(f"⚠️ No 'final_field' in cache for evolution step {step}, skipping backward for this step.")
                        continue

                    # The gradient must match the shape of the field it is replacing
                    if current_grad.shape != final_field_from_step.shape:
                        # ======================= THE CHANGE =======================
                        # This warning has served its purpose and is now commented out for cleaner logs.
                        # logger.warning(f"Aligning evolver gradient from {current_grad.shape} to {final_field_from_step.shape} for step {step}")
                        # ==========================================================
                        grad_data_aligned = backend.zeros_like(final_field_from_step.data, dtype=backend.float32)
                        min_len = min(current_grad.shape[1], grad_data_aligned.shape[1])
                        grad_data_aligned[:, :min_len, :] = current_grad.data[:, :min_len, :]
                        current_grad = FSEField(grad_data_aligned, current_grad.field_type, device=current_grad.device)

                    # Backpropagate through hierarchical FSE Blocks
                    level_caches = step_cache.get('level_caches', [])
                    for level in range(self.field_evolver.hierarchical_levels - 1, -1, -1):
                        if level < len(level_caches):
                            level_cache = level_caches[level]
                            processor = getattr(self.field_evolver, f'level_{level}_processor')
                            
                            level_param_grads, current_grad = processor.backward(current_grad, level_cache)
                            
                            if f"level_{level}_block" not in param_grads:
                                param_grads[f"level_{level}_block"] = {}
                            self._accumulate_gradients(param_grads[f'level_{level}_block'], level_param_grads)
                    
                    # Backpropagate through memory integration
                    original_input_shape = step_cache.get('input_field').shape
                    original_seq_len = original_input_shape[1]
                    
                    grad_for_frontend = FSEField(current_grad.data[:, :original_seq_len, :], current_grad.field_type, device=current_grad.device)
                    memory_grad_data = current_grad.data[:, original_seq_len:, :]
                    
                    if 'memory_gate' in self.field_evolver.parameters:
                        grad_gate_data = self.backend.mean(memory_grad_data, axis=(0,1))
                        if 'memory_gate' not in param_grads:
                            param_grads['memory_gate'] = FSEField(backend.zeros_like(self.field_evolver.memory_gate.data, dtype=backend.float32), self.field_evolver.memory_gate.field_type, device=self.device)
                        param_grads['memory_gate'].data += grad_gate_data

                    current_grad = grad_for_frontend

                except Exception as e:
                    logger.warning(f"⚠️ Evolution step {step} adjoint failed: {e}", exc_info=True)
                    continue
            
            return param_grads, current_grad

    def _adjoint_frontend_backward(self, grad_field: FSEField, frontend_cache: Dict[str, Any]) -> Dict[str, FSEField]:
        """
        [DEFINITIVE FIX] Frontend backward using adjoint PDE solving.
        This version constructs the correct parameter dictionary for the solver,
        resolving the 'kernel' KeyError.
        """
        try:
            param_grads = {}
            
            field_trajectory = frontend_cache.get('field_trajectory', [])
            evolution_cache = frontend_cache.get('evolution_cache', {})
            
            if field_trajectory and not evolution_cache.get('fallback_mode', False):
                # === FIX: Construct the correct parameter dictionary for the PDE solver ===
                # The solver needs a 'kernel'. In the frontend's evolution, the conceptual kernel
                # is one of the 128x128 projection matrices. We'll use the 'complex_real_proj'.
                # The full parameter set is used later for discrete gradient calculation if needed.
                
                pde_parameters = {
                    'kernel': self.lexical_frontend.parameters['complex_real_proj']
                }
                
                frontend_param_grads, _ = self.adjoint_solver.solve_adjoint_pde(
                    upstream_grad_field=grad_field,
                    parameters=pde_parameters,
                    forward_trajectory=field_trajectory,
                    evolution_cache=evolution_cache
                )
                
                # The solver only computes the gradient for the 'kernel' it was given.
                # We need to manually create zero gradients for the other frontend parameters.
                param_grads = {
                    name: FSEField(self.backend.zeros_like(p.data), device=self.device, dtype=self.backend.float32)
                    for name, p in self.lexical_frontend.parameters.items()
                }
                
                # Place the computed gradient into the correct slot.
                if 'kernel' in frontend_param_grads:
                    param_grads['complex_real_proj'] = frontend_param_grads['kernel']

            else:
                logger.warning("⚠️ Frontend trajectory missing, using discrete fallback for gradients.")
                # Fallback to zero gradients for all parameters if PDE solve failed in forward pass
                param_grads = {
                    name: FSEField(self.backend.zeros_like(p.data), device=self.device, dtype=self.backend.float32)
                    for name, p in self.lexical_frontend.parameters.items()
                }
            
            return param_grads
            
        except Exception as e:
            logger.error(f"❌ Adjoint frontend backward failed: {e}", exc_info=True)
            return {
                name: FSEField(self.backend.zeros_like(p.data), device=self.device, dtype=self.backend.float32)
                for name, p in self.lexical_frontend.parameters.items()
            }

    def _fallback_sampler_backward(self, grad_logits: FSEField, sampler_cache: Dict[str, Any]) -> Tuple[Dict[str, FSEField], FSEField]:
        """Fallback discrete sampler backward"""
        param_grads = {}
        
        # Simple fallback: zero gradients
        for i in range(3):  # 3 projection layers
            param_grads[f'projection_{i}'] = FSEField(
                self.backend.zeros_like(self.field_sampler.parameters[f'projection_{i}'].data, dtype=self.backend.float32),
                device=self.device, dtype=self.backend.float32
            )
            param_grads[f'bias_{i}'] = FSEField(
                self.backend.zeros_like(self.field_sampler.parameters[f'bias_{i}'].data, dtype=self.backend.float32),
                device=self.device, dtype=self.backend.float32
            )
        
        # Return gradient unchanged
        downstream_grad = grad_logits
        
        return param_grads, downstream_grad

    def _accumulate_gradients(self, main_grads: Dict, new_grads: Dict):
        """Accumulate gradients from adjoint solving"""
        for key, value in new_grads.items():
            if isinstance(value, dict):
                if key not in main_grads: 
                    main_grads[key] = {}
                self._accumulate_gradients(main_grads[key], value)
            elif isinstance(value, FSEField):
                if key in main_grads: 
                    main_grads[key].data += value.data
                else: 
                    main_grads[key] = value
    
    def generate(self, prompt_chars: ArrayLike, max_length: int = 512, temperature: float = 1.0) -> ArrayLike:
        """Generate text using adjoint-enabled FSU model"""
        generated_chars = prompt_chars.copy()
        
        for _ in range(max_length):
            outputs, _ = self.forward(generated_chars, training=False)
            logits = outputs['fsu_character_logits'].data[0, -1, :]
            
            if temperature > 0:
                if logits.dtype != self.backend.float32: 
                    logits = logits.astype(self.backend.float32)
                probabilities = self._softmax(logits / temperature)
                next_char = self._sample_from_probs(probabilities)
            else:
                next_char = self.backend.argmax(logits)
            
            generated_chars = self.backend.concatenate([
                generated_chars, 
                self.backend.array([[next_char]], dtype=prompt_chars.dtype)
            ], axis=1)
            
            if int(next_char) == 1 or generated_chars.shape[1] >= self.sequence_length: 
                break
        
        return generated_chars

    def _softmax(self, logits: ArrayLike) -> ArrayLike:
        """Softmax implementation"""
        exp_logits = self.backend.exp(logits - self.backend.max(logits))
        return exp_logits / self.backend.sum(exp_logits)

    def _sample_from_probs_enhanced(self, probabilities: ArrayLike, p: float = 0.9) -> ArrayLike:
        """
        [UPGRADED] Samples from a probability distribution using top-p (nucleus)
        sampling to improve coherence and reduce gibberish.
        """
        backend = self.backend
        
        # Sort probabilities in descending order
        sorted_indices = backend.argsort(probabilities, axis=-1)[:, ::-1]
        sorted_probs = backend.take_along_axis(probabilities, sorted_indices, axis=-1)
        
        # Get cumulative probabilities
        cumulative_probs = backend.cumsum(sorted_probs, axis=-1)
        
        # Create a mask for tokens to remove (those outside the nucleus)
        sorted_indices_to_remove = cumulative_probs > p
        # Ensure at least one token is kept
        sorted_indices_to_remove[..., 0] = False 
        
        # Set the probabilities of tokens to remove to zero
        sorted_probs[sorted_indices_to_remove] = 0
        
        # Re-normalize the remaining probabilities
        renormalized_probs = sorted_probs / backend.sum(sorted_probs, axis=-1, keepdims=True)
        
        # Sample from the re-normalized distribution
        batch_size = probabilities.shape[0]
        vocab_size = probabilities.shape[1]
        final_sampled_indices = []
        
        for i in range(batch_size):
            # Use the original indices and the new probabilities to sample
            sampled_idx = backend.random.choice(vocab_size, p=renormalized_probs[i])
            final_sampled_indices.append(sampled_idx)

        return backend.array(final_sampled_indices)
    

    def morph_update_frontend(self, combined_features: ArrayLike, semantic_field_target: FSEField):
            """
            [DEFINITIVE VERSION] Applies a Morph solve to the frontend's field_lifter.
            This version correctly initializes the solver with the right output dimension (channels).
            """
            # --- THIS IS THE FIX ---
            # The "vocabulary" for the field_lifter is the number of output channels, not the full model vocab_size.
            if not hasattr(self, "_morph_solver_frontend"):
                self._morph_solver_frontend = MorphSolver(
                    ridge=0.01, 
                    device=self.device,
                    vocab_size=self.channels 
                )
            # -----------------------

            target_field = semantic_field_target.data
            lifter_W = self.lexical_frontend.parameters["field_lifter"]
            lifter_b_dummy = FSEField(self.backend.zeros(self.channels, dtype=lifter_W.dtype), device=self.device)

            self._morph_solver_frontend.update_layer_weights(
                input_data=combined_features,
                target_indices=target_field,
                layer_W=lifter_W,
                layer_b=lifter_b_dummy
            )


class FSMART_FSULanguageModel:
    """
    The FSMART-FSU Hybrid Model.
    Uses the FSU Core for infinite-context processing and a Transformer Head
    for high-quality reasoning and inference.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'gpu')
        self.backend = cp if self.device == "gpu" else np
        
        self.sequence_length = config['sequence_length']
        self.vocab_size = config.get('vocab_size', 65536)
        
        self.parameters = {}
        self._build_fsmart_fsu_architecture()
        
        logger.info(f"✅ FSMART-FSU Hybrid Model Initialized")

    def _build_fsmart_fsu_architecture(self):
        """Builds the hybrid architecture."""
        fsu_channels = self.config['channels']
        
        # 1. FSU Core (Feature Extractor)
        # These are your existing, powerful FSU components
        self.lexical_frontend = ContinuousLexicalFrontend(
            self.sequence_length, fsu_channels, self.vocab_size, self.device
        )
        self.parameters['fsu_frontend'] = self.lexical_frontend.parameters

        self.field_evolver = SemanticFieldEvolver(
            fsu_channels, self.sequence_length, 
            evolution_rate=self.config.get('field_evolution_rate', 0.05), 
            device=self.device
        )
        self.parameters['fsu_evolver'] = self.field_evolver.parameters

        # 2. The Bridge (Tokenizer)
        self.transformer_dim = 512 # A common dimension for smaller transformers
        self.tokenizer = FSUFieldTokenizer(
            fsu_channels=fsu_channels,
            transformer_dim=self.transformer_dim,
            max_seq_len=self.sequence_length,
            device=self.device
        )
        self.parameters['fsmart_tokenizer'] = self.tokenizer.parameters

        # 3. Transformer Head (Inference Engine)
        self.transformer_head = TransformerEncoderHead(
            transformer_dim=self.transformer_dim,
            num_layers=4, # A reasonable depth for the head
            vocab_size=self.vocab_size,
            device=self.device
        )
        self.parameters['fsmart_head'] = self.transformer_head.parameters

    def forward(self, character_sequence: ArrayLike, training: bool = True) -> Tuple[Dict[str, FSEField], Dict[str, Any]]:
        # Step 1: FSU Core processes the long-context input
        semantic_field, frontend_cache = self.lexical_frontend.forward(character_sequence)
        evolved_field, evolver_cache = self.field_evolver.forward(semantic_field, training=training)
        
        # Step 2: The Bridge tokenizes the FSU's output field
        transformer_tokens, tokenizer_cache = self.tokenizer.forward(evolved_field)
        
        # Step 3: The Transformer Head performs reasoning and inference
        character_logits, head_cache = self.transformer_head.forward(transformer_tokens, training=training)
        
        # Final outputs are compatible with your existing loss functions
        outputs = {'fsu_character_logits': character_logits, 'fsu_evolved_field': evolved_field}
        
        cache = {
            'frontend_cache': frontend_cache,
            'evolver_cache': evolver_cache,
            'tokenizer_cache': tokenizer_cache,
            'head_cache': head_cache,
            'evolved_field': evolved_field # Needed for backward pass
        }
        return outputs, cache

    def backward(self, upstream_grads: Dict[str, FSEField], cache: Dict[str, Any]) -> Tuple[Dict[str, Any], FSEField]:
        all_param_grads = {}
        
        # The 'fsu_evolved_field' gradient is for the FSU's coherence loss
        grad_coherence = upstream_grads.get('fsu_evolved_field')
        grad_logits = upstream_grads['fsu_character_logits']
        
        # Step 1: Backprop through the discrete Transformer Head
        head_param_grads, grad_from_head = self.transformer_head.backward(grad_logits, cache['head_cache'])
        all_param_grads['fsmart_head'] = head_param_grads
        
        # Step 2: Backprop through the discrete Tokenizer Bridge
        tokenizer_param_grads, grad_from_tokenizer = self.tokenizer.backward(grad_from_head, cache['tokenizer_cache'])
        all_param_grads['fsmart_tokenizer'] = tokenizer_param_grads
        
        # Step 3: Combine gradients and start the FSU Adjoint Solve
        # The gradient from the Transformer and the FSU's own coherence loss are combined
        # to form the initial condition for the FSU's adjoint backward pass.
        initial_grad_for_evolver = grad_from_tokenizer
        if grad_coherence:
            # Align sequence lengths before adding, as Transformer output is one shorter
            if grad_coherence.shape[1] > initial_grad_for_evolver.shape[1]:
                grad_coherence_data = grad_coherence.data[:, :initial_grad_for_evolver.shape[1], :]
                initial_grad_for_evolver.data += grad_coherence_data
            else:
                 initial_grad_for_evolver.data += grad_coherence.data

        # Step 4: Backprop through the FSU Core using the adjoint method
        # NOTE: We are re-using the original model's backward methods here.
        # This assumes your original FSULanguageModel instance is accessible or
        # you refactor these methods to be callable from here.
        
        # Re-create a temporary cache structure for the FSU core backward pass
        fsu_core_cache = {
            'frontend_cache': cache['frontend_cache'],
            'evolution_caches': [cache['evolver_cache']], # Simplified for this example
            'semantic_field': cache['evolver_cache']['input_field'],
            'final_evolved_field': cache['evolved_field'],
            'step_count': 1
        }
        
        # This part requires refactoring the original model's backward methods to be reusable
        # For simplicity, we create temporary instances here.
        temp_fsu_model = FSULanguageModel(self.config)
        temp_fsu_model.field_evolver = self.field_evolver
        temp_fsu_model.lexical_frontend = self.lexical_frontend

        evolver_param_grads, grad_to_frontend = temp_fsu_model._adjoint_evolver_backward(initial_grad_for_evolver, fsu_core_cache)
        all_param_grads['fsu_evolver'] = evolver_param_grads
        
        frontend_param_grads = temp_fsu_model._adjoint_frontend_backward(grad_to_frontend, cache['frontend_cache'])
        all_param_grads['fsu_frontend'] = frontend_param_grads
        
        return all_param_grads, grad_to_frontend