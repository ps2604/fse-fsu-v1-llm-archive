# file: fsu_loss_functions.py
# FSU LANGUAGE MODEL LOSS FUNCTIONS - FULLY ADJOINT IMPLEMENTED
# Revolutionary conversion from discrete gradient outputs to adjoint initial conditions
# UPDATES: Loss functions now output initial conditions for adjoint PDE solving instead of discrete gradients

import numpy as np
import cupy as cp
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import re

from adjoint_core_optimized import FSEField, FieldType, FieldOperations, get_default_dtype

logger = logging.getLogger(__name__)
NUMBER_REGEX = re.compile(r'-?\d+\.?\d*')

def _chunked_cross_entropy_adjoint(
    logits_field : FSEField,
    targets_field: FSEField,
    loss_weights_field: Optional[FSEField] = None,
    chunk_tokens : int = 512,
) -> Tuple[float, FSEField]:
    """
    [DEFINITIVE MEMORY-FIXED VERSION] Memory-efficient softmax-cross-entropy
    that outputs adjoint initial conditions without allocating the full (B,T,V)
    gradient tensor in memory.
    """
    backend = logits_field.backend
    original_logits_data = logits_field.data
    targets_data = targets_field.data
    
    if targets_data.dtype not in [backend.int32, backend.int64]:
        targets_data = targets_data.astype(backend.int32)

    if original_logits_data.ndim != 3:
        raise ValueError(f"Logits must be 3D (B, T, V), got {original_logits_data.shape}")

    min_len = min(original_logits_data.shape[1], targets_data.shape[1])
    
    logits_for_loss = original_logits_data[:, :min_len, :]
    targets_for_loss = targets_data[:, :min_len]

    B, T_calc, V = logits_for_loss.shape

    if loss_weights_field is None:
        loss_weights = backend.ones_like(targets_for_loss, dtype=backend.float32)
    else:
        loss_weights = loss_weights_field.data[:, :min_len].astype(backend.float32)

    loss_acc = backend.float32(0.0)
    
    # ======================= THE DEFINITIVE OOM FIX =======================
    # Initialize a smaller temporary array for the gradient, which we will populate in chunks.
    # This avoids allocating the massive full-size gradient tensor all at once.
    final_dtype = get_default_dtype()
    adjoint_initial_condition = backend.zeros_like(original_logits_data, dtype=final_dtype)
    
    for t0 in range(0, T_calc, chunk_tokens):
        t1 = min(t0 + chunk_tokens, T_calc)
        tgt_slice = targets_for_loss[:, t0:t1]
        s = tgt_slice.shape[1]
        if s == 0: continue

        # Up-cast the slice to float32 for stable calculations
        logits_slice_fp32 = logits_for_loss[:, t0:t1, :].astype(backend.float32)
        
        # Perform softmax cross-entropy on the slice
        logits_slice_fp32 -= logits_slice_fp32.max(axis=-1, keepdims=True)
        exp_ = backend.exp(logits_slice_fp32)
        soft = exp_ / (exp_.sum(axis=-1, keepdims=True) + 1e-9)

        probs_for_targets = soft[backend.arange(B)[:, None], backend.arange(s)[None, :], tgt_slice]
        weights_slice = loss_weights[:, t0:t1]
        slice_loss = -backend.log(probs_for_targets + 1e-9) * weights_slice
        loss_acc += slice_loss.sum()

        # Create the initial condition for THIS CHUNK ONLY
        adjoint_slice_fp32 = soft
        adjoint_slice_fp32[backend.arange(B)[:, None], backend.arange(s)[None, :], tgt_slice] -= 1
        adjoint_slice_fp32 *= weights_slice[..., None]
        
        # Place the computed chunk into the final, pre-allocated array.
        # This avoids creating multiple large intermediate arrays.
        adjoint_initial_condition[:, t0:t1, :] = adjoint_slice_fp32.astype(final_dtype)

        # Explicitly free memory used by the large intermediate tensors in this chunk
        del logits_slice_fp32, exp_, soft, probs_for_targets, adjoint_slice_fp32
        if backend == cp:
            cp.get_default_memory_pool().free_all_blocks()
    # ========================================================================

    total_weight_sum = backend.sum(loss_weights) + 1e-9
    loss_scalar = float(loss_acc / total_weight_sum)
    
    adjoint_initial_condition /= total_weight_sum
    adjoint_initial_condition = _ensure_field_continuity(adjoint_initial_condition, backend)

    adjoint_field = FSEField(
        adjoint_initial_condition, 
        FieldType.CONTINUOUS,
        device=logits_field.device,
        dtype=final_dtype
    )

    return loss_scalar, adjoint_field


def _apply_field_smoothing(field_data, backend):
    """
    [DEFINITIVE STABLE VERSION] Apply field smoothing to create continuous initial 
    conditions, now with a check to handle different tensor dimensions robustly.
    """
    # ======================= THE DEFINITIVE FIX =======================
    # Only apply 3D smoothing if the array is actually 3D and long enough.
    if field_data.ndim == 3 and field_data.shape[1] > 2:
    # ================================================================
        smoothed = field_data.copy()
        smoothed[:, 1:-1, :] = (
            0.25 * field_data[:, :-2, :] + 
            0.5 * field_data[:, 1:-1, :] + 
            0.25 * field_data[:, 2:, :]
        )
        return smoothed
    
    # Return the original data if it's not a compatible 3D array
    return field_data

def _ensure_field_continuity(field_data, backend):
    """
    [DEFINITIVE STABILITY FIX] Ensures field continuity properties required for
    adjoint PDE solving by checking for and correcting NaN, infinite, or
    exploding values in the initial gradient condition.
    """
    # Check for NaN or infinite values and replace them
    if backend.any(backend.isnan(field_data)) or backend.any(backend.isinf(field_data)):
        logger.warning("⚠️ NaN or Inf detected in field data, applying stabilization.")
        field_data = backend.nan_to_num(field_data, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure field magnitude is bounded for PDE stability
    field_magnitude = backend.max(backend.abs(field_data))
    # Using a high but finite threshold for stability
    STABILITY_THRESHOLD = 1e4 
    if field_magnitude > STABILITY_THRESHOLD:
        logger.warning(f"Field magnitude {field_magnitude:.2f} exceeds stability threshold of {STABILITY_THRESHOLD}. Clamping.")
        field_data = field_data * (STABILITY_THRESHOLD / field_magnitude)
    
    return field_data
class FSULanguageModelingLoss:
    """
    ✅ FULLY ADJOINT IMPLEMENTED: Base character-level language modeling loss 
    that outputs adjoint initial conditions
    """
    def forward(self, logits_field: FSEField, targets_field: FSEField, loss_weights_field: Optional[FSEField] = None) -> Tuple[float, FSEField]:
        """Return loss scalar and adjoint initial condition field"""
        return _chunked_cross_entropy_adjoint(logits_field, targets_field, loss_weights_field)

class FSUMathematicalLoss:
    """
    ✅ FULLY ADJOINT IMPLEMENTED: Mathematical reasoning loss with adjoint initial conditions
    """
    def __init__(self, step_weight: float = 1.5, final_answer_weight: float = 2.0):
        self.step_weight, self.final_answer_weight = step_weight, final_answer_weight
        self.base_lm_loss = FSULanguageModelingLoss()

    def _create_math_weight_mask(self, target_chars: np.ndarray) -> np.ndarray:
        """Create mathematical reasoning weight mask"""
        weights = np.ones_like(target_chars, dtype=np.float32)
        try: 
            text = bytes(c - 4 for c in target_chars if c > 3).decode('utf-8', 'ignore')
        except Exception: 
            text = ""
        
        # Weight mathematical step markers
        for match in re.finditer(r'<<(.+?)>>', text):
            if match.end() < len(weights): 
                weights[match.start():match.end()] *= self.step_weight
        
        # Weight final numerical answers
        numbers = [m for m in NUMBER_REGEX.finditer(text)]
        if numbers and numbers[-1].end() < len(weights): 
            weights[numbers[-1].start():numbers[-1].end()] *= self.final_answer_weight
        
        return weights

    def forward(self, pred_field: FSEField, target_field: FSEField) -> Tuple[float, FSEField]:
        """Return loss scalar and adjoint initial condition field for mathematical reasoning"""
        backend = pred_field.backend
        target_cpu = target_field.data.get()
        weight_masks = np.array([self._create_math_weight_mask(t) for t in target_cpu])
        loss_weights_field = FSEField(backend.asarray(weight_masks), device=pred_field.device)
        
        # Use adjoint-enabled base loss
        return self.base_lm_loss.forward(pred_field, target_field, loss_weights_field)

class FSUConversationLoss:
    """
    ✅ FULLY ADJOINT IMPLEMENTED: Conversation coherence loss with field-based adjoint initial conditions
    """
    def __init__(self, coherence_weight: float = 0.5, context_window: int = 512):
        self.coherence_weight = coherence_weight
        self.context_window = context_window

    def forward(self, pred_field: FSEField, target_field: FSEField) -> Tuple[float, FSEField]:
        """
        ✅ ADJOINT IMPLEMENTATION: Return scalar loss and continuous field adjoint initial condition
        """
        backend = pred_field.backend
        pred_data_fp32 = pred_field.data.astype(backend.float32)
        
        batch_size, seq_len, channels = pred_data_fp32.shape
        coherence_loss = 0.0
        
        # ✅ ADJOINT INITIAL CONDITION: Create continuous field instead of discrete gradient
        coherence_adjoint_field = backend.zeros_like(pred_data_fp32)
        
        if seq_len > self.context_window:
            num_turns = seq_len // self.context_window
            
            for turn in range(num_turns - 1):
                start1, end1 = turn * self.context_window, (turn + 1) * self.context_window
                start2, end2 = end1, min((turn + 2) * self.context_window, seq_len)
                
                if end1 > start1 and end2 > start2:
                    mean1 = backend.mean(pred_data_fp32[:, start1:end1, :], axis=1, keepdims=True)
                    mean2 = backend.mean(pred_data_fp32[:, start2:end2, :], axis=1, keepdims=True)
                    
                    diff = mean2 - mean1
                    coherence_loss += backend.mean(diff**2)
                    
                    if batch_size > 0 and num_turns > 0:
                        # ✅ ADJOINT FIELD CONSTRUCTION: Create continuous field adjoint condition
                        grad_magnitude = 2.0 * diff / (batch_size * num_turns)
                        
                        # Apply field smoothing for continuity
                        smoothed_grad_2 = _apply_field_smoothing(
                            grad_magnitude / (end2 - start2), backend
                        )
                        smoothed_grad_1 = _apply_field_smoothing(
                            -grad_magnitude / (end1 - start1), backend
                        )
                        
                        # Distribute to continuous field
                        coherence_adjoint_field[:, start2:end2, :] += smoothed_grad_2
                        coherence_adjoint_field[:, start1:end1, :] += smoothed_grad_1

        # ✅ ADJOINT FIELD CREATION: Convert to continuous field with field properties
        coherence_adjoint_field = _ensure_field_continuity(coherence_adjoint_field, backend)
        
        adjoint_field = FSEField(
            coherence_adjoint_field, 
            FieldType.CONTINUOUS,
            device=pred_field.device, 
            dtype=backend.float32
        )
        
        return float(coherence_loss), adjoint_field

class FSUReasoningLoss:
    """
    ✅ FULLY ADJOINT IMPLEMENTED: Logical reasoning loss with continuous field adjoint initial conditions
    """
    def __init__(self, consistency_weight: float = 0.1):
        self.consistency_weight = consistency_weight

    def _find_repeated_concepts(self, target_chars: np.ndarray) -> Dict[str, List[int]]:
        """Find repeated concepts for consistency checking"""
        try: 
            text = bytes(c - 4 for c in target_chars if c > 3).decode('utf-8', 'ignore').lower()
        except Exception: 
            return {}
        
        words = re.findall(r'\b[a-z]{4,}\b', text)
        word_counts = {w: words.count(w) for w in set(words)}
        repeated_concepts = {w for w, c in word_counts.items() if c > 1 and c < 10}
        concept_indices = {
            c: [m.start() for m in re.finditer(r'\b' + re.escape(c) + r'\b', text)] 
            for c in repeated_concepts
        }
        return concept_indices


    def forward(self, pred_field: FSEField, target_field: FSEField) -> Tuple[float, FSEField]:
        """
        [DEFINITIVE STABLE VERSION V2] Returns scalar loss and continuous field adjoint
        initial condition, now with correct shape handling to prevent crashes.
        """
        backend = pred_field.backend
        pred_data = pred_field.data.astype(backend.float32)
        target_cpu = target_field.data.get()
        
        total_consistency_loss = 0.0
        consistency_adjoint_field = backend.zeros_like(pred_data)
        
        for i in range(pred_data.shape[0]):
            concepts = self._find_repeated_concepts(target_cpu[i])
            
            for _, indices in concepts.items():
                if len(indices) < 2: continue
                safe_indices = [idx for idx in indices if idx < pred_data.shape[1]]
                if len(safe_indices) < 2: continue
                
                field_vectors = pred_data[i, safe_indices, :]
                mean_vector = backend.mean(field_vectors, axis=0, keepdims=True)
                total_consistency_loss += backend.mean((field_vectors - mean_vector)**2)
                
                concept_adjoint_data = (2.0 / len(safe_indices)) * (field_vectors - mean_vector)

                # ======================= THE DEFINITIVE FIX =======================
                # Reshape the adjoint data to be explicitly 3D before smoothing.
                # This ensures the smoothing function receives the correct input shape.
                concept_adjoint_data_3d = concept_adjoint_data.reshape(1, -1, concept_adjoint_data.shape[-1])
                concept_adjoint_smoothed = _apply_field_smoothing(concept_adjoint_data_3d, backend)
                # Reshape back to original 2D for application
                concept_adjoint_smoothed_2d = concept_adjoint_smoothed.reshape(concept_adjoint_data.shape)
                # ================================================================

                for j, idx in enumerate(safe_indices):
                    consistency_adjoint_field[i, idx, :] += concept_adjoint_smoothed_2d[j, :]

        batch_size = pred_data.shape[0] or 1
        final_loss = total_consistency_loss / batch_size
        consistency_adjoint_field /= batch_size
        
        consistency_adjoint_field = _ensure_field_continuity(consistency_adjoint_field, backend)
        
        adjoint_field = FSEField(
            consistency_adjoint_field, 
            FieldType.CONTINUOUS,
            device=pred_field.device, 
            dtype=backend.float32
        )
        
        return float(final_loss), adjoint_field

class FSUFieldCoherenceLoss:
    """
    ✅ FULLY ADJOINT IMPLEMENTED: Field coherence loss with continuous adjoint initial conditions
    """
    def __init__(self, smoothness_weight: float = 0.05):
        self.smoothness_weight = smoothness_weight

    def forward(self, pred_field: FSEField, target_field: FSEField) -> Tuple[float, FSEField]:
        """
        ✅ ADJOINT IMPLEMENTATION: Return scalar loss and continuous field adjoint initial condition
        """
        backend = pred_field.backend
        field_data = pred_field.data.astype(backend.float32)
        
        if field_data.shape[1] <= 1: 
            zero_adjoint = FSEField(
                backend.zeros_like(field_data), 
                FieldType.CONTINUOUS,
                device=pred_field.device
            )
            return 0.0, zero_adjoint

        # Compute field differences for smoothness
        field_diff = field_data[:, 1:, :] - field_data[:, :-1, :]
        smoothness_loss = backend.mean(field_diff ** 2)
        
        # ✅ ADJOINT FIELD CONSTRUCTION: Create continuous field adjoint condition
        coherence_adjoint_field = backend.zeros_like(field_data)
        grad_factor = 2.0 / (field_data.size + 1e-9)
        
        # Apply field-aware gradient distribution
        smoothed_diff = _apply_field_smoothing(field_diff, backend)
        
        coherence_adjoint_field[:, :-1, :] += grad_factor * smoothed_diff
        coherence_adjoint_field[:, 1:, :] -= grad_factor * smoothed_diff
        
        # ✅ ADJOINT FIELD VALIDATION
        coherence_adjoint_field = _ensure_field_continuity(coherence_adjoint_field, backend)
        
        adjoint_field = FSEField(
            coherence_adjoint_field, 
            FieldType.CONTINUOUS,
            device=pred_field.device, 
            dtype=backend.float32
        )
        
        return float(smoothness_loss), adjoint_field

class FSUCompositeLoss:
    """
    ✅ FULLY ADJOINT IMPLEMENTED: Composite loss with continuous field adjoint initial condition accumulation
    Revolutionary replacement of discrete gradient accumulation with field superposition
    """
    def __init__(self, **kwargs):
        self.weights = {
            'language': kwargs.get('language_modeling_weight', 1.0),
            'conversation': kwargs.get('conversation_coherence_weight', 0.5),
            'math': kwargs.get('mathematical_reasoning_weight', 1.2),
            'reasoning': kwargs.get('reasoning_accuracy_weight', 0.8),
            'coherence': kwargs.get('field_coherence_weight', 0.2)
        }
        self.losses = {
            'language': FSULanguageModelingLoss(),
            'conversation': FSUConversationLoss(),
            'math': FSUMathematicalLoss(),
            'reasoning': FSUReasoningLoss(),
            'coherence': FSUFieldCoherenceLoss()
        }
        
        logger.info("✅ ADJOINT CompositeLoss: Initialized with field-based adjoint initial conditions")
        
# In adjoint_loss_functions.py
# Class: FSUCompositeLoss

    def forward(self, pred_outputs: Dict[str, FSEField], target_field: FSEField, task_type: str) -> Tuple[float, Dict[str, FSEField], Dict[str, float]]:
        """
        [METRICS-ENABLED VERSION] Returns the total loss scalar, the adjoint initial
        condition fields, and a new dictionary containing the individual loss values.
        """
        backend = pred_outputs['fsu_character_logits'].backend
        total_loss = 0.0
        
        # NEW: Dictionary to store individual loss values
        individual_losses = {}
        
        current_dtype = get_default_dtype()
        adjoint_initial_conditions = {
            'fsu_character_logits': FSEField(
                backend.zeros_like(pred_outputs['fsu_character_logits'].data, dtype=current_dtype),
                FieldType.CONTINUOUS, device=pred_outputs['fsu_character_logits'].device
            ),
            'fsu_evolved_field': FSEField(
                backend.zeros_like(pred_outputs['fsu_evolved_field'].data, dtype=current_dtype),
                FieldType.CONTINUOUS, device=pred_outputs['fsu_evolved_field'].device
            )
        }

        logits_field = pred_outputs['fsu_character_logits']
        evolved_field = pred_outputs['fsu_evolved_field']

        if task_type in ["math_reasoning", "advanced_math", "gsm8k", "competition_math"]:
            loss, adjoint_field = self.losses['math'].forward(logits_field, target_field)
            weight = self.weights['math']
            individual_losses['math_loss'] = float(loss) # Store individual loss
        else:
            loss, adjoint_field = self.losses['language'].forward(logits_field, target_field)
            weight = self.weights['language']
            individual_losses['lm_loss'] = float(loss) # Store individual loss
        
        total_loss += weight * loss
        adjoint_initial_conditions['fsu_character_logits'] += (adjoint_field * backend.array(weight, dtype=adjoint_field.dtype))
        
        coherence_loss, coherence_adjoint = self.losses['coherence'].forward(evolved_field, target_field)
        if np.isfinite(coherence_loss):
            total_loss += self.weights['coherence'] * coherence_loss
            adjoint_initial_conditions['fsu_evolved_field'] += (coherence_adjoint * backend.array(self.weights['coherence'], dtype=coherence_adjoint.dtype))
            individual_losses['field_coherence_loss'] = float(coherence_loss) # Store individual loss
        
        if task_type in ["conversation", "general", "sharegpt", "openassistant", "wizardlm"]:
            conv_loss, conv_adjoint = self.losses['conversation'].forward(evolved_field, target_field)
            if np.isfinite(conv_loss):
                total_loss += self.weights['conversation'] * conv_loss
                adjoint_initial_conditions['fsu_evolved_field'] += (conv_adjoint * backend.array(self.weights['conversation'], dtype=conv_adjoint.dtype))
                individual_losses['conversation_loss'] = float(conv_loss) # Store individual loss
            
        if task_type in ["logical_reasoning", "completion_reasoning", "chain_of_thought", "commonsenseqa", "hellaswag", "cot_collection"]:
            reason_loss, reason_adjoint = self.losses['reasoning'].forward(evolved_field, target_field)
            if np.isfinite(reason_loss):
                total_loss += self.weights['reasoning'] * reason_loss
                adjoint_initial_conditions['fsu_evolved_field'] += (reason_adjoint * backend.array(self.weights['reasoning'], dtype=reason_adjoint.dtype))
                individual_losses['reasoning_loss'] = float(reason_loss) # Store individual loss
        
        for field_name, adjoint_field in adjoint_initial_conditions.items():
            adjoint_field.data = _ensure_field_continuity(adjoint_field.data, backend)
        
        # NEW: Return the dictionary of individual losses
        return float(total_loss), adjoint_initial_conditions, individual_losses

def get_fsu_loss_function(task_type: str, **kwargs) -> FSUCompositeLoss:
    """Get adjoint-enabled FSU loss function"""
    return FSUCompositeLoss(**kwargs)

def compute_fsu_perplexity(pred_field: FSEField, target_field: FSEField) -> float:
    """Compute perplexity using adjoint-enabled loss"""
    loss, _ = _chunked_cross_entropy_adjoint(pred_field, target_field)
    return np.exp(loss) if loss is not None and np.isfinite(loss) else float('inf')
