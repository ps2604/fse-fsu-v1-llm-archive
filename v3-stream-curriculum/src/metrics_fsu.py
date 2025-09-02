# file: metrics_fsu.py
# COMPREHENSIVE FSU METRICS: Language processing metrics for Field-Sequence Unit architecture
# MIXED PRECISION OPTIMIZED: fp16→fp32 conversion for accuracy, memory optimization, FSU field coherence
# UPDATES: Mixed precision handling, memory optimization, enhanced field coherence metrics, fp32 accuracy

import numpy as np
import cupy as cp
from typing import Dict, Any, Optional, Union, List
import logging
import re
import math
import gc
from collections import defaultdict

from adjoint_core_optimized import FSEField, FieldType, get_default_dtype

logger = logging.getLogger(__name__)

def _sequence_grad(x):
    """
    Forward-difference gradients for 1D semantic sequences [B,S,C].
    Returns dx with the same shape for sequence direction.
    Works for NumPy or CuPy arrays with mixed precision optimization.
    """
    lib = cp if isinstance(x, cp.ndarray) else np
    dx = lib.zeros_like(x)

    # sequence-direction (S axis)
    dx[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]

    return dx

class FSUMetricsComputer:
    """Comprehensive FSU metrics computer for language processing tasks with mixed precision optimization"""
    
    def __init__(self, device: str = "gpu", vocab_size: int = 65536, use_mixed_precision: bool = True):
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.vocab_size = vocab_size
        self.use_mixed_precision = use_mixed_precision
        
        # ✅ MIXED PRECISION: Get current training precision and computation precision
        self.training_dtype = get_default_dtype() if use_mixed_precision else self.backend.float32
        self.computation_dtype = self.backend.float32  # Always compute metrics in fp32 for accuracy
        
        logger.debug(f"✅ FSU Metrics Computer initialized: device={device}, "
                    f"training_dtype={self.training_dtype}, computation_dtype={self.computation_dtype}")
        
    def _ensure_fp32_for_metrics(self, field_or_array: Union[FSEField, np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        """
        ✅ MIXED PRECISION: Convert FSEField or array to fp32 for accurate metric computation
        """
        if isinstance(field_or_array, FSEField):
            data = field_or_array.data
        else:
            data = field_or_array
        
        # Convert to fp32 if not already
        if data.dtype != self.computation_dtype:
            return data.astype(self.computation_dtype, copy=False)
        return data
    
    def _cleanup_intermediate_tensors(self, *tensors):
        """
        ✅ MEMORY OPTIMIZATION: Clean up intermediate tensors to prevent memory accumulation
        """
        try:
            for tensor in tensors:
                if hasattr(tensor, 'data'):
                    del tensor.data
                del tensor
            
            if self.device == "gpu":
                # Force garbage collection for GPU memory
                gc.collect()
        except Exception as e:
            logger.debug(f"Tensor cleanup warning: {e}")
    
    def compute_all_metrics(self, predictions: Dict[str, FSEField], targets: Dict[str, FSEField], 
                           generated_text: Optional[str] = None, target_text: Optional[str] = None,
                           conversation_context: Optional[List[str]] = None) -> Dict[str, float]:
        """
        [DEFINITIVE FIX V2] Compute all comprehensive FSU language metrics.
        This version standardizes the check for language modeling keys and adds
        explicit logging to confirm when perplexity is being calculated.
        """
        metrics = {}
        
        try:
            # === DEFINITIVE FIX: Standardized Language Modeling Metrics Check ===
            # We now ONLY check for the officially supported keys. This removes ambiguity.
            if 'fsu_character_logits' in predictions and 'fsu_character_targets' in targets:
                logger.debug("Found 'fsu_character_logits' and 'fsu_character_targets'. Computing perplexity...")
                lm_metrics = self.compute_language_modeling_metrics(
                    predictions['fsu_character_logits'], targets['fsu_character_targets']
                )
                metrics.update(lm_metrics)
            else:
                # This log will tell you exactly why perplexity is being skipped.
                logger.warning(
                    f"Skipping perplexity calculation. Keys found in predictions: {list(predictions.keys())}. "
                    f"Keys found in targets: {list(targets.keys())}."
                )

            # ✅ ENHANCED: Field coherence metrics for semantic processing with mixed precision
            # This part remains the same.
            semantic_fields = [key for key in predictions.keys() if 'semantic' in key or 'field' in key]
            for field_key in semantic_fields:
                if field_key in predictions:
                    field_metrics = self.compute_semantic_field_metrics(
                        predictions[field_key], targets.get(field_key)
                    )
                    # Add field-specific prefix to avoid naming conflicts
                    field_specific_metrics = {f"{field_key}_{k}": v for k, v in field_metrics.items()}
                    metrics.update(field_specific_metrics)
            
            # Fallback for semantic field if specific keys not found
            if not semantic_fields and 'semantic_field' in predictions:
                field_metrics = self.compute_semantic_field_metrics(
                    predictions['semantic_field'], targets.get('semantic_field')
                )
                metrics.update(field_metrics)
            
            # This part remains the same.
            if generated_text and target_text:
                conv_metrics = self.compute_conversation_metrics(
                    generated_text, target_text, conversation_context
                )
                metrics.update(conv_metrics)
            
            # This part remains the same.
            reasoning_fields = [key for key in predictions.keys() if 'reasoning' in key]
            for reasoning_key in reasoning_fields:
                if reasoning_key in predictions and reasoning_key in targets:
                    reasoning_metrics = self.compute_reasoning_metrics(
                        predictions[reasoning_key], targets[reasoning_key],
                        generated_text, target_text
                    )
                    metrics.update(reasoning_metrics)
            
            # This part remains the same.
            global_metrics = self.compute_global_fsu_metrics(predictions, targets)
            metrics.update(global_metrics)
            
            # ✅ MEMORY OPTIMIZATION: Cleanup intermediate computation data
            if self.device == "gpu" and len(metrics) > 10:
                gc.collect()
            
        except Exception as e:
            logger.debug(f"FSU metrics computation failed: {e}")
            metrics = {'fsu_computation_error': 1.0}
            
        return metrics
    
    def compute_language_modeling_metrics(
            self,
            pred_logits: FSEField,
            target_chars: FSEField
        ) -> Dict[str, float]:
            """
            [DEFINITIVE STABLE VERSION V3] Computes all language modeling metrics.
            This final version aligns both perplexity and accuracy calculations to only
            evaluate on real text characters, ignoring all special/padding tokens.
            """
            metrics: Dict[str, float] = {}
            epsilon = 1e-9

            try:
                # --- Data and Shape Preparation ---
                if pred_logits.device != target_chars.device:
                    target_chars = target_chars.to_device(pred_logits.device)

                logits_fp32 = self._ensure_fp32_for_metrics(pred_logits)
                target_idx = target_chars.data.astype(self.backend.int32)
                
                if target_idx.ndim == 3 and target_idx.shape[-1] == 1:
                    target_idx = target_idx.squeeze(-1)

                if logits_fp32.shape[:2] != target_idx.shape:
                    logger.error(f"Unrecoverable shape mismatch: logits {logits_fp32.shape[:2]} vs targets {target_idx.shape}")
                    return {'fsu_perplexity': float('inf'), 'fsu_character_accuracy': 0.0}

                # --- NaN/Inf Guard ---
                if self.backend.isnan(logits_fp32).any() or self.backend.isinf(logits_fp32).any():
                    logger.error("[METRICS] Logits contain NaN/Inf – skipping metrics for this batch")
                    return {'fsu_perplexity': float('inf'), 'fsu_character_accuracy': 0.0}

                B, T, V = logits_fp32.shape

                # ======================= THE DEFINITIVE FIX =======================
                # This new, stricter mask ignores ALL special tokens (PAD, S, E, UNK)
                # by only considering token IDs >= 4.
                # Both perplexity and accuracy will now be calculated only on real characters.
                
                valid_text_mask = (target_idx >= 4)
                
                if self.backend.sum(valid_text_mask) == 0:
                    # This batch contains no real characters to evaluate, return neutral metrics.
                    return {'fsu_cross_entropy': float(self.backend.log(V)), 'fsu_perplexity': float(V), 'fsu_character_accuracy': 0.0}
                
                # --- NLL & Perplexity Calculation (on real characters only) ---
                max_l = self.backend.max(logits_fp32, axis=-1, keepdims=True)
                exp_shifted = self.backend.exp(logits_fp32 - max_l)
                sum_exp = self.backend.sum(exp_shifted, axis=-1, keepdims=True)
                probs = exp_shifted / (sum_exp + epsilon)

                batch_idx = self.backend.arange(B)[:, None]
                time_idx = self.backend.arange(T)[None, :]
                gold_probs = probs[batch_idx, time_idx, target_idx]

                # Calculate NLL using the valid_text_mask
                nll = -self.backend.mean(self.backend.log(gold_probs[valid_text_mask] + epsilon))
                metrics['fsu_cross_entropy'] = float(nll)
                metrics['fsu_perplexity'] = math.exp(min(float(nll), 70.0))

                # --- Accuracy Calculation (on real characters only) ---
                pred_chars = self.backend.argmax(logits_fp32, axis=-1)
                correct = (pred_chars == target_idx) & valid_text_mask
                acc = self.backend.sum(correct.astype(self.computation_dtype)) / self.backend.sum(valid_text_mask.astype(self.computation_dtype))
                metrics['fsu_character_accuracy'] = float(acc)
                # ================================================================

            except Exception as e:
                logger.error(f"Language modeling metrics failed: {e}", exc_info=True)
                metrics['fsu_perplexity'] = float('inf')
                metrics['fsu_character_accuracy'] = 0.0

            return metrics

    
    def compute_semantic_field_metrics(self, pred_field: FSEField, target_field: Optional[FSEField] = None) -> Dict[str, float]:
        """Compute semantic field coherence and stability metrics with mixed precision optimization"""
        metrics = {}
        
        try:
            # ✅ MIXED PRECISION: Convert to fp32 for accurate field analysis
            pred_field_fp32 = self._ensure_fp32_for_metrics(pred_field)
            
            # ✅ ENHANCED: Field coherence - measure smoothness of semantic transitions
            coherence = self.compute_fse_sequence_coherence_fp32(pred_field_fp32)
            metrics['fsu_semantic_field_coherence'] = coherence
            
            # ✅ MIXED PRECISION: Field stability - measure of field evolution stability
            field_grad = _sequence_grad(pred_field_fp32)
            grad_magnitude = self.backend.sqrt(self.backend.sum(field_grad ** 2, axis=-1))
            stability = float(1.0 / (1.0 + self.backend.mean(grad_magnitude)))
            metrics['fsu_field_stability'] = stability
            
            # ✅ MIXED PRECISION: Field complexity - measure of semantic richness
            field_variance = self.backend.var(pred_field_fp32, axis=-1)
            complexity = float(self.backend.mean(field_variance))
            metrics['fsu_semantic_complexity'] = complexity
            
            # ✅ ENHANCED: Context length scaling - field quality vs sequence length
            if pred_field_fp32.shape[1] > 1000:  # Long sequence
                try:
                    early_coherence = self.compute_fse_sequence_coherence_fp32(pred_field_fp32[:, :1000])
                    late_coherence = self.compute_fse_sequence_coherence_fp32(pred_field_fp32[:, -1000:])
                    coherence_degradation = float(early_coherence - late_coherence)
                    metrics['fsu_long_context_degradation'] = coherence_degradation
                except Exception as e:
                    logger.debug(f"Long context analysis failed: {e}")
                    metrics['fsu_long_context_degradation'] = 0.0
            
            # ✅ MIXED PRECISION: Field magnitude analysis
            field_magnitude = self.backend.sqrt(self.backend.sum(pred_field_fp32 ** 2, axis=-1))
            avg_magnitude = float(self.backend.mean(field_magnitude))
            max_magnitude = float(self.backend.max(field_magnitude))
            metrics['fsu_field_avg_magnitude'] = avg_magnitude
            metrics['fsu_field_max_magnitude'] = max_magnitude
            
            # ✅ ENHANCED: Target field similarity if available
            if target_field is not None:
                try:
                    if target_field.device != pred_field.device:
                        target_field = target_field.to_device(pred_field.device)
                    
                    target_field_fp32 = self._ensure_fp32_for_metrics(target_field)
                    
                    # Cosine similarity between semantic fields
                    field_similarity = self._cosine_similarity(pred_field_fp32, target_field_fp32)
                    metrics['fsu_semantic_field_similarity'] = float(field_similarity)
                    
                    # Mean squared error between fields
                    mse = self.backend.mean((pred_field_fp32 - target_field_fp32) ** 2)
                    metrics['fsu_field_mse'] = float(mse)
                    
                except Exception as e:
                    logger.debug(f"Target field similarity computation failed: {e}")
            
            # ✅ MEMORY OPTIMIZATION: Cleanup intermediate tensors
            self._cleanup_intermediate_tensors(field_grad, grad_magnitude, field_variance, field_magnitude)
            
        except Exception as e:
            logger.debug(f"Semantic field metrics failed: {e}")
            metrics = {'fsu_semantic_field_coherence': 0.5, 'fsu_field_stability': 0.5}
            
        return metrics
    
    def compute_conversation_metrics(self, generated_text: str, target_text: str, 
                                   conversation_context: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute conversation quality and coherence metrics with enhanced analysis"""
        metrics = {}
        
        try:
            # ✅ ENHANCED: BLEU score for generation quality
            bleu_score = self._compute_bleu_score(generated_text, target_text)
            metrics['fsu_bleu_score'] = bleu_score
            
            # ✅ ENHANCED: Response relevance with multiple measures
            relevance = self._compute_response_relevance(generated_text, target_text)
            metrics['fsu_response_relevance'] = relevance
            
            # ✅ ENHANCED: Response length appropriateness
            gen_len = len(generated_text)
            target_len = len(target_text)
            length_ratio = min(gen_len, target_len) / max(gen_len, target_len, 1)
            metrics['fsu_length_appropriateness'] = length_ratio
            
            # ✅ ENHANCED: Conversation coherence (if context provided)
            if conversation_context:
                coherence = self._compute_conversation_coherence(
                    generated_text, conversation_context
                )
                metrics['fsu_conversation_coherence'] = coherence
                
                # Context utilization
                context_usage = self._compute_context_utilization(
                    generated_text, conversation_context
                )
                metrics['fsu_context_utilization'] = context_usage
            
            # ✅ ENHANCED: Response fluency with multiple indicators
            fluency = self._compute_text_fluency(generated_text)
            metrics['fsu_response_fluency'] = fluency
            
            # ✅ NEW: Response diversity measurement
            diversity = self._compute_response_diversity(generated_text)
            metrics['fsu_response_diversity'] = diversity
            
            # ✅ NEW: Semantic consistency measurement
            consistency = self._compute_semantic_consistency(generated_text, target_text)
            metrics['fsu_semantic_consistency'] = consistency
            
        except Exception as e:
            logger.debug(f"Conversation metrics failed: {e}")
            metrics = {'fsu_bleu_score': 0.0, 'fsu_response_relevance': 0.0}
            
        return metrics
    
    def compute_reasoning_metrics(self, pred_reasoning: FSEField, target_reasoning: FSEField,
                                generated_text: str, target_text: str) -> Dict[str, float]:
        """Compute reasoning quality metrics with mixed precision optimization"""
        metrics = {}
        
        try:
            # ✅ MIXED PRECISION: Convert reasoning fields to fp32
            pred_reasoning_fp32 = self._ensure_fp32_for_metrics(pred_reasoning)
            target_reasoning_fp32 = self._ensure_fp32_for_metrics(target_reasoning)
            
            # ✅ ENHANCED: Mathematical accuracy (for math problems)
            if self._contains_math(target_text):
                math_accuracy = self._evaluate_mathematical_accuracy(generated_text, target_text)
                metrics['fsu_mathematical_accuracy'] = math_accuracy
                
                # Step-by-step reasoning quality
                step_quality = self._evaluate_reasoning_steps(generated_text, target_text)
                metrics['fsu_reasoning_step_quality'] = step_quality
            
            # ✅ ENHANCED: Logical consistency
            logical_consistency = self._evaluate_logical_consistency(generated_text)
            metrics['fsu_logical_consistency'] = logical_consistency
            
            # ✅ ENHANCED: Chain-of-thought coherence
            if any(word in generated_text.lower() for word in ["because", "therefore", "thus", "since"]):
                cot_coherence = self._evaluate_cot_coherence(generated_text)
                metrics['fsu_chain_of_thought_coherence'] = cot_coherence
            
            # ✅ MIXED PRECISION: Reasoning field coherence in fp32
            if pred_reasoning.device != target_reasoning.device:
                target_reasoning = target_reasoning.to_device(pred_reasoning.device)
            
            reasoning_coherence = self.compute_fse_sequence_coherence_fp32(pred_reasoning_fp32)
            metrics['fsu_reasoning_field_coherence'] = reasoning_coherence
            
            # ✅ MIXED PRECISION: Multi-step reasoning stability
            reasoning_stability = self._compute_reasoning_stability(pred_reasoning_fp32)
            metrics['fsu_multi_step_reasoning_stability'] = reasoning_stability
            
            # ✅ ENHANCED: Field similarity between prediction and target
            field_similarity = self._cosine_similarity(pred_reasoning_fp32, target_reasoning_fp32)
            metrics['fsu_reasoning_field_similarity'] = float(field_similarity)
            
        except Exception as e:
            logger.debug(f"Reasoning metrics failed: {e}")
            metrics = {'fsu_logical_consistency': 0.5, 'fsu_reasoning_field_coherence': 0.5}
            
        return metrics
    
    def compute_fse_sequence_coherence_fp32(self, field_data: Union[np.ndarray, cp.ndarray]) -> float:
        """
        ✅ MIXED PRECISION: Compute FSE coherence for 1D semantic sequences in fp32
        """
        try:
            # Ensure fp32 computation
            field_data = field_data.astype(self.computation_dtype, copy=False)
            
            # Compute sequence gradient
            dx = _sequence_grad(field_data)
            
            # Magnitude of semantic changes
            mag = self.backend.sqrt(self.backend.sum(dx**2, axis=-1))
            
            # Coherence is inverse of gradient magnitude variation
            grad_variation = self.backend.std(mag)
            coherence = float(1.0 / (1.0 + grad_variation))
            
            return coherence
        except Exception as e:
            logger.debug(f"Sequence coherence computation failed: {e}")
            return 0.5
    
    def compute_global_fsu_metrics(self, predictions: Dict[str, FSEField], targets: Dict[str, FSEField]) -> Dict[str, float]:
        """Compute global FSU system metrics with mixed precision optimization"""
        metrics = {}
        
        try:
            # ✅ MIXED PRECISION: Overall field coherence across all modalities in fp32
            all_coherences = []
            for field_name, field in predictions.items():
                if isinstance(field, FSEField) and len(field.shape) >= 2:
                    field_fp32 = self._ensure_fp32_for_metrics(field)
                    coherence = self.compute_fse_sequence_coherence_fp32(field_fp32)
                    all_coherences.append(coherence)
            
            if all_coherences:
                global_coherence = float(self.backend.mean(self.backend.array(all_coherences)))
                metrics['global_fsu_coherence'] = global_coherence
            
            # ✅ ENHANCED: System prediction consistency with mixed precision
            if len(predictions) > 1:
                field_consistency = self._compute_cross_field_consistency(predictions)
                metrics['global_field_consistency'] = field_consistency
            
            # ✅ ENHANCED: FSU computational efficiency (if step count info available)
            if any('step' in key for key in predictions.keys()):
                efficiency = self._compute_fsu_efficiency(predictions)
                metrics['fsu_computational_efficiency'] = efficiency
            
            # ✅ NEW: Global field magnitude statistics
            all_magnitudes = []
            for field_name, field in predictions.items():
                if isinstance(field, FSEField) and len(field.shape) >= 2:
                    field_fp32 = self._ensure_fp32_for_metrics(field)
                    magnitude = self.backend.sqrt(self.backend.sum(field_fp32 ** 2, axis=-1))
                    all_magnitudes.append(float(self.backend.mean(magnitude)))
            
            if all_magnitudes:
                metrics['global_field_avg_magnitude'] = float(np.mean(all_magnitudes))
                metrics['global_field_magnitude_std'] = float(np.std(all_magnitudes))
            
        except Exception as e:
            logger.debug(f"Global FSU metrics failed: {e}")
            metrics = {'global_fsu_coherence': 0.5}
            
        return metrics
    
    # ✅ MIXED PRECISION: Helper methods for metric computation with fp32 accuracy
    def _log_softmax(self, logits, eps: float = 1e-9):
        """
        Numerically-stable log-softmax implemented in fp32.

        Args
        ----
        logits : ndarray [B,T,V]  (NumPy or CuPy)
        eps    : small constant to avoid log(0) when V==1
        """
        logits = logits.astype(self.computation_dtype, copy=False)
        max_l  = self.backend.max(logits, axis=-1, keepdims=True)
        shifted = logits - max_l
        log_sum = self.backend.log(
            self.backend.sum(self.backend.exp(shifted), axis=-1, keepdims=True) + eps
        )
        return shifted - log_sum

    def _softmax(self, logits, eps: float = 1e-9):
        """Stable soft-max wrapper used by probability metrics."""
        logits = logits.astype(self.computation_dtype, copy=False)
        max_l  = self.backend.max(logits, axis=-1, keepdims=True)
        exp_l  = self.backend.exp(logits - max_l)
        return exp_l / (self.backend.sum(exp_l, axis=-1, keepdims=True) + eps)
    
    def _gather_target_log_probs(self, log_probs, target_indices):
        """Gather log probabilities for target characters with proper shape handling"""
        if len(log_probs.shape) == 3 and len(target_indices.shape) == 2:
            batch_size, seq_length, vocab_size = log_probs.shape
            flat_indices = target_indices.flatten()
            batch_indices = self.backend.repeat(self.backend.arange(batch_size), seq_length)
            seq_indices = self.backend.tile(self.backend.arange(seq_length), batch_size)
            
            # Ensure indices are within bounds
            flat_indices = self.backend.clip(flat_indices, 0, vocab_size - 1)
            
            flat_log_probs = log_probs[batch_indices, seq_indices, flat_indices]
            return flat_log_probs.reshape(batch_size, seq_length)
        else:
            # Handle different shapes gracefully
            target_indices_clipped = self.backend.clip(target_indices, 0, log_probs.shape[-1] - 1)
            if len(log_probs.shape) == 2 and len(target_indices.shape) == 1:
                return log_probs[self.backend.arange(len(target_indices)), target_indices_clipped]
            else:
                # Fallback for unexpected shapes
                return self.backend.zeros_like(target_indices, dtype=self.computation_dtype)
    
    def _compute_topk_accuracy(self, logits, targets, valid_mask, k=5):
        """Compute top-k accuracy with mixed precision"""
        logits = logits.astype(self.computation_dtype, copy=False)
        
        if logits.shape[-1] < k:
            k = logits.shape[-1]
        
        # Get top-k predictions
        top_k_indices = self.backend.argsort(logits, axis=-1)[..., -k:]
        
        # Expand targets for comparison
        targets_expanded = self.backend.expand_dims(targets, axis=-1)
        
        # Check if target is in top-k
        correct = self.backend.any(top_k_indices == targets_expanded, axis=-1)
        correct_masked = correct & valid_mask
        
        return float(self.backend.sum(correct_masked.astype(self.computation_dtype)) / 
                    self.backend.sum(valid_mask.astype(self.computation_dtype)))
    
    def _cosine_similarity(self, a, b):
        """Compute cosine similarity between field arrays in fp32"""
        a = a.astype(self.computation_dtype, copy=False)
        b = b.astype(self.computation_dtype, copy=False)
        
        # Flatten to compute similarity
        a_flat = a.reshape(-1, a.shape[-1])
        b_flat = b.reshape(-1, b.shape[-1])
        
        # Compute cosine similarity
        dot_product = self.backend.sum(a_flat * b_flat, axis=-1)
        norm_a = self.backend.linalg.norm(a_flat, axis=-1)
        norm_b = self.backend.linalg.norm(b_flat, axis=-1)
        
        # Avoid division by zero
        similarity = dot_product / (norm_a * norm_b + 1e-8)
        return float(self.backend.mean(similarity))
    
    def _compute_bleu_score(self, generated: str, reference: str) -> float:
        """Enhanced BLEU score computation with n-gram analysis"""
        try:
            gen_words = generated.lower().split()
            ref_words = reference.lower().split()
            
            if not gen_words or not ref_words:
                return 0.0
            
            # BLEU-1 through BLEU-4 calculation
            bleu_scores = []
            
            for n in range(1, 5):  # 1-gram to 4-gram
                # Generate n-grams
                gen_ngrams = [tuple(gen_words[i:i+n]) for i in range(len(gen_words)-n+1)]
                ref_ngrams = [tuple(ref_words[i:i+n]) for i in range(len(ref_words)-n+1)]
                
                if not gen_ngrams or not ref_ngrams:
                    bleu_scores.append(0.0)
                    continue
                
                # Count matches
                gen_ngram_count = defaultdict(int)
                ref_ngram_count = defaultdict(int)
                
                for ngram in gen_ngrams:
                    gen_ngram_count[ngram] += 1
                for ngram in ref_ngrams:
                    ref_ngram_count[ngram] += 1
                
                matches = 0
                for ngram, count in gen_ngram_count.items():
                    matches += min(count, ref_ngram_count[ngram])
                
                precision = matches / len(gen_ngrams) if gen_ngrams else 0
                bleu_scores.append(precision)
            
            # Weighted average (BLEU-4 style)
            weights = [0.25, 0.25, 0.25, 0.25]
            weighted_score = sum(w * s for w, s in zip(weights, bleu_scores))
            
            # Brevity penalty
            bp = min(1.0, len(gen_words) / len(ref_words)) if ref_words else 0
            
            return weighted_score * bp
        except Exception as e:
            logger.debug(f"BLEU score computation failed: {e}")
            return 0.0
    
    def _compute_response_relevance(self, generated: str, target: str) -> float:
        """Enhanced semantic relevance computation"""
        try:
            # Word overlap
            gen_words = set(generated.lower().split())
            target_words = set(target.lower().split())
            
            if not target_words:
                return 0.0
            
            # Basic word overlap
            overlap = len(gen_words.intersection(target_words))
            word_relevance = overlap / len(target_words)
            
            # Character-level similarity (for handling morphological variations)
            char_overlap = 0
            gen_chars = set(generated.lower())
            target_chars = set(target.lower())
            char_overlap = len(gen_chars.intersection(target_chars)) / len(target_chars) if target_chars else 0
            
            # Combine word and character relevance
            relevance = 0.7 * word_relevance + 0.3 * char_overlap
            return min(1.0, relevance)
        except Exception as e:
            logger.debug(f"Response relevance computation failed: {e}")
            return 0.0
    
    def _compute_conversation_coherence(self, response: str, context: List[str]) -> float:
        """Enhanced conversation coherence with temporal analysis"""
        try:
            if not context:
                return 1.0
            
            response_words = set(response.lower().split())
            
            # Weight recent context more heavily
            coherence_scores = []
            for i, turn in enumerate(context[-5:]):  # Last 5 turns
                context_words = set(turn.lower().split())
                if context_words:
                    overlap = len(response_words.intersection(context_words))
                    turn_coherence = overlap / len(context_words)
                    
                    # Weight more recent turns more heavily
                    weight = (i + 1) / len(context[-5:])
                    coherence_scores.append(turn_coherence * weight)
            
            return float(np.mean(coherence_scores)) if coherence_scores else 0.5
        except Exception as e:
            logger.debug(f"Conversation coherence computation failed: {e}")
            return 0.5
    
    def _compute_context_utilization(self, response: str, context: List[str]) -> float:
        """Enhanced context utilization measurement"""
        try:
            if not context:
                return 1.0
            
            response_lower = response.lower()
            utilization_score = 0.0
            
            # Check for explicit references
            reference_patterns = [
                'you said', 'you mentioned', 'earlier', 'before', 'previously',
                'as you noted', 'you asked', 'your question', 'your point'
            ]
            
            for pattern in reference_patterns:
                if pattern in response_lower:
                    utilization_score += 0.2
            
            # Check for topic continuity with recent context
            recent_context = ' '.join(context[-3:]).lower()
            context_words = set(recent_context.split())
            response_words = set(response_lower.split())
            
            if context_words and response_words:
                overlap_ratio = len(context_words.intersection(response_words)) / len(context_words)
                utilization_score += min(0.6, overlap_ratio)
            
            # Check for coherent topic development
            if len(context) > 1:
                topic_development = self._analyze_topic_development(response, context)
                utilization_score += topic_development * 0.2
            
            return min(1.0, utilization_score)
        except Exception as e:
            logger.debug(f"Context utilization computation failed: {e}")
            return 0.5
    
    def _compute_text_fluency(self, text: str) -> float:
        """Enhanced fluency metric with multiple linguistic indicators"""
        try:
            if not text:
                return 0.0
            
            words = text.split()
            if not words:
                return 0.0
            
            fluency_scores = []
            
            # 1. Average word length (reasonable range)
            avg_word_len = sum(len(word) for word in words) / len(words)
            word_len_score = 1.0 - abs(avg_word_len - 5.0) / 10.0  # Optimal around 5 chars
            fluency_scores.append(max(0.0, word_len_score))
            
            # 2. Sentence structure analysis
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            if sentences:
                words_per_sentence = len(words) / len(sentences)
                sentence_score = 1.0 - abs(words_per_sentence - 15.0) / 20.0  # Optimal around 15 words
                fluency_scores.append(max(0.0, sentence_score))
            
            # 3. Vocabulary diversity
            unique_words = len(set(word.lower() for word in words))
            diversity_score = unique_words / len(words) if words else 0
            fluency_scores.append(diversity_score)
            
            # 4. Grammatical structure indicators
            grammar_score = self._assess_grammatical_structure(text)
            fluency_scores.append(grammar_score)
            
            # 5. Repetition penalty
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word.lower()] += 1
            
            max_freq = max(word_freq.values()) if word_freq else 1
            repetition_penalty = 1.0 - min(0.5, (max_freq - 1) / len(words))
            fluency_scores.append(repetition_penalty)
            
            return float(np.mean(fluency_scores))
        except Exception as e:
            logger.debug(f"Fluency computation failed: {e}")
            return 0.5
    
    def _compute_response_diversity(self, text: str) -> float:
        """Measure response diversity and creativity"""
        try:
            words = text.lower().split()
            if len(words) < 2:
                return 0.0
            
            # Lexical diversity
            unique_words = len(set(words))
            lexical_diversity = unique_words / len(words)
            
            # Bigram diversity
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            unique_bigrams = len(set(bigrams))
            bigram_diversity = unique_bigrams / len(bigrams) if bigrams else 0
            
            # Average word rarity (approximated by length)
            avg_word_rarity = sum(min(1.0, len(word) / 8.0) for word in words) / len(words)
            
            diversity_score = (lexical_diversity + bigram_diversity + avg_word_rarity) / 3.0
            return min(1.0, diversity_score)
        except Exception as e:
            logger.debug(f"Response diversity computation failed: {e}")
            return 0.5
    
    def _compute_semantic_consistency(self, generated: str, target: str) -> float:
        """Measure semantic consistency between generated and target"""
        try:
            # Extract key concepts from both texts
            gen_concepts = self._extract_key_concepts(generated)
            target_concepts = self._extract_key_concepts(target)
            
            if not target_concepts:
                return 1.0 if not gen_concepts else 0.5
            
            # Concept overlap
            concept_overlap = len(gen_concepts.intersection(target_concepts)) / len(target_concepts)
            
            # Structural similarity (sentence count, question marks, etc.)
            gen_structure = self._analyze_text_structure(generated)
            target_structure = self._analyze_text_structure(target)
            
            structure_similarity = 1.0 - abs(gen_structure['sentence_ratio'] - target_structure['sentence_ratio'])
            structure_similarity *= 1.0 - abs(gen_structure['question_ratio'] - target_structure['question_ratio'])
            
            consistency = (concept_overlap + structure_similarity) / 2.0
            return min(1.0, consistency)
        except Exception as e:
            logger.debug(f"Semantic consistency computation failed: {e}")
            return 0.5
    
    def _contains_math(self, text: str) -> bool:
        """Check if text contains mathematical content"""
        math_indicators = ['+', '-', '*', '/', '=', 'solve', 'calculate', 'equation', 'answer is', 
                          'result', 'sum', 'product', 'divide', 'multiply', '∑', '∫', '∂']
        return any(indicator in text.lower() for indicator in math_indicators)
    
    def _evaluate_mathematical_accuracy(self, generated: str, target: str) -> float:
        """Enhanced mathematical accuracy evaluation"""
        try:
            # Extract numerical answers
            gen_numbers = re.findall(r'-?\d+\.?\d*', generated)
            target_numbers = re.findall(r'-?\d+\.?\d*', target)
            
            if not target_numbers:
                return 1.0 if not gen_numbers else 0.0
            
            accuracy_scores = []
            
            # Check final numerical answers
            if gen_numbers and target_numbers:
                try:
                    gen_final = float(gen_numbers[-1])
                    target_final = float(target_numbers[-1])
                    
                    # Allow small numerical errors
                    relative_error = abs(gen_final - target_final) / (abs(target_final) + 1e-8)
                    accuracy_scores.append(1.0 if relative_error < 0.01 else 0.0)
                except ValueError:
                    accuracy_scores.append(0.0)
            
            # Check intermediate steps if available
            if len(gen_numbers) > 1 and len(target_numbers) > 1:
                intermediate_accuracy = sum(
                    1.0 for gn, tn in zip(gen_numbers[:-1], target_numbers[:-1])
                    if abs(float(gn) - float(tn)) < 0.1
                ) / max(len(gen_numbers[:-1]), len(target_numbers[:-1]))
                accuracy_scores.append(intermediate_accuracy)
            
            # Check mathematical terminology consistency
            math_terms = ['sum', 'product', 'difference', 'quotient', 'equals', 'result']
            gen_terms = [term for term in math_terms if term in generated.lower()]
            target_terms = [term for term in math_terms if term in target.lower()]
            
            if target_terms:
                term_accuracy = len(set(gen_terms).intersection(set(target_terms))) / len(target_terms)
                accuracy_scores.append(term_accuracy)
            
            return float(np.mean(accuracy_scores)) if accuracy_scores else 0.0
        except Exception as e:
            logger.debug(f"Mathematical accuracy evaluation failed: {e}")
            return 0.0
    
    def _evaluate_reasoning_steps(self, generated: str, target: str) -> float:
        """Enhanced reasoning step quality evaluation"""
        try:
            # Enhanced reasoning indicators
            reasoning_words = [
                'because', 'since', 'therefore', 'thus', 'so', 'then', 'first', 'next', 'finally',
                'given that', 'it follows', 'we can conclude', 'this means', 'as a result',
                'step 1', 'step 2', 'step 3', 'initially', 'subsequently'
            ]
            
            gen_reasoning = sum(1 for word in reasoning_words if word in generated.lower())
            target_reasoning = sum(1 for word in reasoning_words if word in target.lower())
            
            if target_reasoning == 0:
                return 1.0 if gen_reasoning > 0 else 0.5
            
            # Step structure analysis
            gen_steps = len(re.findall(r'step \d+|first|then|next|finally', generated.lower()))
            target_steps = len(re.findall(r'step \d+|first|then|next|finally', target.lower()))
            
            step_quality_scores = []
            
            # Reasoning indicator ratio
            indicator_ratio = min(1.0, gen_reasoning / target_reasoning)
            step_quality_scores.append(indicator_ratio)
            
            # Step structure similarity
            if target_steps > 0:
                step_structure_score = min(1.0, gen_steps / target_steps)
                step_quality_scores.append(step_structure_score)
            
            # Logical flow assessment
            logical_flow = self._assess_logical_flow(generated)
            step_quality_scores.append(logical_flow)
            
            return float(np.mean(step_quality_scores))
        except Exception as e:
            logger.debug(f"Reasoning steps evaluation failed: {e}")
            return 0.5
    
    def _evaluate_logical_consistency(self, text: str) -> float:
        """Enhanced logical consistency evaluation"""
        try:
            text_lower = text.lower()
            consistency_score = 1.0
            
            # Enhanced contradiction detection
            contradictions = [
                ('yes', 'no'), ('true', 'false'), ('always', 'never'), ('all', 'none'),
                ('increase', 'decrease'), ('positive', 'negative'), ('correct', 'incorrect'),
                ('possible', 'impossible'), ('certain', 'uncertain'), ('agree', 'disagree')
            ]
            
            for pos, neg in contradictions:
                if pos in text_lower and neg in text_lower:
                    # Check context distance to avoid false positives
                    pos_idx = text_lower.find(pos)
                    neg_idx = text_lower.find(neg)
                    if abs(pos_idx - neg_idx) < 100:  # Closer = more likely contradiction
                        consistency_score -= 0.1
            
            # Check for logical connectors consistency
            logical_connectors = ['however', 'but', 'although', 'despite', 'nevertheless']
            contradiction_count = sum(1 for connector in logical_connectors if connector in text_lower)
            
            # Moderate contradictions are okay, excessive ones indicate inconsistency
            if contradiction_count > len(text.split()) / 20:  # More than 5% contradiction words
                consistency_score -= 0.2
            
            # Temporal consistency check
            temporal_consistency = self._check_temporal_consistency(text)
            consistency_score *= temporal_consistency
            
            return max(0.0, consistency_score)
        except Exception as e:
            logger.debug(f"Logical consistency evaluation failed: {e}")
            return 0.5
    
    def _evaluate_cot_coherence(self, text: str) -> float:
        """Enhanced chain-of-thought coherence evaluation"""
        try:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            if len(sentences) <= 1:
                return 0.5
            
            coherence_scores = []
            
            # Flow indicator analysis
            flow_indicators = [
                'first', 'then', 'next', 'finally', 'because', 'since', 'therefore',
                'thus', 'so', 'as a result', 'this means', 'given that'
            ]
            
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                has_flow = any(indicator in sentence_lower for indicator in flow_indicators)
                
                if i == 0:  # First sentence doesn't need flow indicators
                    coherence_scores.append(1.0)
                elif has_flow:
                    coherence_scores.append(1.0)
                else:
                    # Check for implicit flow (topic continuity)
                    if i > 0:
                        prev_words = set(sentences[i-1].lower().split())
                        curr_words = set(sentence_lower.split())
                        word_overlap = len(prev_words.intersection(curr_words))
                        implicit_flow = min(1.0, word_overlap / 5.0)  # Up to 5 shared words = good flow
                        coherence_scores.append(implicit_flow)
                    else:
                        coherence_scores.append(0.5)
            
            # Penalize abrupt topic changes
            topic_consistency = self._assess_topic_consistency_within_text(text)
            
            avg_coherence = float(np.mean(coherence_scores))
            return avg_coherence * topic_consistency
        except Exception as e:
            logger.debug(f"CoT coherence evaluation failed: {e}")
            return 0.5
    
    def _compute_reasoning_stability(self, reasoning_field: Union[np.ndarray, cp.ndarray]) -> float:
        """Enhanced reasoning field stability with mixed precision"""
        try:
            reasoning_field = reasoning_field.astype(self.computation_dtype, copy=False)
            
            # Measure field variation during reasoning
            field_grad = _sequence_grad(reasoning_field)
            grad_variance = self.backend.var(field_grad)
            
            # Additional stability metrics
            field_magnitude = self.backend.sqrt(self.backend.sum(reasoning_field ** 2, axis=-1))
            magnitude_stability = 1.0 / (1.0 + self.backend.std(field_magnitude))
            
            # Combine gradient and magnitude stability
            gradient_stability = 1.0 / (1.0 + grad_variance)
            
            overall_stability = (float(gradient_stability) + float(magnitude_stability)) / 2.0
            return overall_stability
        except Exception as e:
            logger.debug(f"Reasoning stability computation failed: {e}")
            return 0.5
    
    def _compute_cross_field_consistency(self, predictions: Dict[str, FSEField]) -> float:
        """Enhanced cross-field consistency with mixed precision"""
        try:
            # Extract field activations in fp32
            activations = []
            field_similarities = []
            
            field_list = []
            for field_name, field in predictions.items():
                if isinstance(field, FSEField) and len(field.shape) >= 2:
                    field_fp32 = self._ensure_fp32_for_metrics(field)
                    activation = float(self.backend.mean(self.backend.abs(field_fp32)))
                    activations.append(activation)
                    field_list.append((field_name, field_fp32))
            
            if len(activations) <= 1:
                return 1.0
            
            # Activation consistency
            activation_std = float(np.std(activations))
            activation_mean = float(np.mean(activations))
            activation_consistency = 1.0 / (1.0 + activation_std / (activation_mean + 1e-7))
            
            # Cross-field similarity analysis
            for i, (name1, field1) in enumerate(field_list):
                for j, (name2, field2) in enumerate(field_list[i+1:], i+1):
                    if field1.shape == field2.shape:
                        similarity = self._cosine_similarity(field1, field2)
                        field_similarities.append(similarity)
            
            # Combine metrics
            if field_similarities:
                avg_similarity = float(np.mean(field_similarities))
                similarity_consistency = 1.0 - float(np.std(field_similarities))
                
                overall_consistency = (activation_consistency + avg_similarity + similarity_consistency) / 3.0
            else:
                overall_consistency = activation_consistency
            
            return max(0.0, min(1.0, overall_consistency))
        except Exception as e:
            logger.debug(f"Cross-field consistency computation failed: {e}")
            return 0.5
    
    def _compute_fsu_efficiency(self, predictions: Dict[str, FSEField]) -> float:
        """Enhanced FSU computational efficiency measurement"""
        try:
            total_operations = 0
            total_elements = 0
            field_sparsities = []
            
            for field_name, field in predictions.items():
                if isinstance(field, FSEField):
                    field_fp32 = self._ensure_fp32_for_metrics(field)
                    elements = field_fp32.size
                    
                    # Estimate operations based on non-zero elements and field activity
                    non_zero = self.backend.count_nonzero(self.backend.abs(field_fp32) > 1e-6)
                    activity_level = float(self.backend.mean(self.backend.abs(field_fp32)))
                    
                    # Consider field-specific operation costs
                    if 'semantic' in field_name:
                        operation_multiplier = 2.0  # Semantic fields require more processing
                    elif 'reasoning' in field_name:
                        operation_multiplier = 1.5  # Reasoning fields are moderately complex
                    else:
                        operation_multiplier = 1.0
                    
                    estimated_ops = non_zero * operation_multiplier * (1.0 + activity_level)
                    total_operations += estimated_ops
                    total_elements += elements
                    
                    # Field sparsity
                    sparsity = 1.0 - (non_zero / elements)
                    field_sparsities.append(sparsity)
            
            if total_elements > 0:
                # Efficiency based on sparsity and operation density
                operation_efficiency = 1.0 - (total_operations / total_elements)
                sparsity_efficiency = float(np.mean(field_sparsities)) if field_sparsities else 0.0
                
                overall_efficiency = (operation_efficiency + sparsity_efficiency) / 2.0
                return max(0.0, min(1.0, overall_efficiency))
            
            return 1.0
        except Exception as e:
            logger.debug(f"FSU efficiency computation failed: {e}")
            return 0.5
    
    # ✅ NEW: Additional helper methods for enhanced analysis
    def _extract_key_concepts(self, text: str) -> set:
        """Extract key concepts from text"""
        try:
            words = text.lower().split()
            # Filter out common words and extract meaningful concepts
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'}
            concepts = {word for word in words if len(word) > 2 and word not in stopwords}
            return concepts
        except Exception:
            return set()
    
    def _analyze_text_structure(self, text: str) -> dict:
        """Analyze structural properties of text"""
        try:
            sentences = len(re.findall(r'[.!?]+', text))
            questions = len(re.findall(r'\?', text))
            words = len(text.split())
            
            return {
                'sentence_ratio': sentences / max(1, words / 10),  # sentences per 10 words
                'question_ratio': questions / max(1, sentences),   # questions per sentence
                'avg_sentence_length': words / max(1, sentences)
            }
        except Exception:
            return {'sentence_ratio': 0.0, 'question_ratio': 0.0, 'avg_sentence_length': 0.0}
    
    def _assess_grammatical_structure(self, text: str) -> float:
        """Simple grammatical structure assessment"""
        try:
            # Check for basic grammatical patterns
            score = 0.0
            
            # Capitalization at sentence starts
            sentences = re.split(r'[.!?]+', text)
            properly_capitalized = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
            if sentences:
                score += (properly_capitalized / len(sentences)) * 0.3
            
            # Punctuation usage
            has_periods = '.' in text
            has_appropriate_spacing = '  ' not in text  # No double spaces
            
            if has_periods:
                score += 0.3
            if has_appropriate_spacing:
                score += 0.2
            
            # Reasonable sentence length variance
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            if sentence_lengths:
                length_variance = np.std(sentence_lengths)
                if 2 <= length_variance <= 8:  # Reasonable variance
                    score += 0.2
            
            return min(1.0, score)
        except Exception:
            return 0.5
    
    def _analyze_topic_development(self, response: str, context: List[str]) -> float:
        """Analyze how well response develops topics from context"""
        try:
            if not context:
                return 0.5
            
            # Extract topics from recent context
            recent_topics = set()
            for turn in context[-2:]:
                recent_topics.update(self._extract_key_concepts(turn))
            
            # Extract topics from response
            response_topics = self._extract_key_concepts(response)
            
            if not recent_topics:
                return 0.5
            
            # Measure topic development
            topic_continuation = len(recent_topics.intersection(response_topics)) / len(recent_topics)
            new_topic_introduction = len(response_topics - recent_topics) / max(1, len(response_topics))
            
            # Good balance between continuation and development
            development_score = topic_continuation * 0.7 + min(0.3, new_topic_introduction)
            return development_score
        except Exception:
            return 0.5
    
    def _check_temporal_consistency(self, text: str) -> float:
        """Check for temporal consistency in text"""
        try:
            # Look for temporal indicators
            past_indicators = ['was', 'were', 'had', 'did', 'yesterday', 'before', 'earlier', 'previously']
            present_indicators = ['is', 'are', 'have', 'do', 'now', 'currently', 'today']
            future_indicators = ['will', 'shall', 'going to', 'tomorrow', 'later', 'next', 'soon']
            
            text_lower = text.lower()
            
            past_count = sum(1 for ind in past_indicators if ind in text_lower)
            present_count = sum(1 for ind in present_indicators if ind in text_lower)
            future_count = sum(1 for ind in future_indicators if ind in text_lower)
            
            total_temporal = past_count + present_count + future_count
            
            if total_temporal == 0:
                return 1.0  # No temporal indicators = no temporal inconsistency
            
            # Check for reasonable temporal distribution
            max_temporal = max(past_count, present_count, future_count)
            temporal_consistency = 1.0 - (max_temporal / total_temporal - 0.6)  # Penalize if one tense dominates too much
            
            return max(0.0, min(1.0, temporal_consistency))
        except Exception:
            return 1.0
    
    def _assess_topic_consistency_within_text(self, text: str) -> float:
        """Assess topic consistency within a single text"""
        try:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            if len(sentences) <= 1:
                return 1.0
            
            # Extract concepts from each sentence
            sentence_concepts = []
            for sentence in sentences:
                concepts = self._extract_key_concepts(sentence)
                sentence_concepts.append(concepts)
            
            # Measure overlap between consecutive sentences
            consistency_scores = []
            for i in range(len(sentence_concepts) - 1):
                current_concepts = sentence_concepts[i]
                next_concepts = sentence_concepts[i + 1]
                
                if current_concepts and next_concepts:
                    overlap = len(current_concepts.intersection(next_concepts))
                    union = len(current_concepts.union(next_concepts))
                    jaccard_similarity = overlap / union if union > 0 else 0
                    consistency_scores.append(jaccard_similarity)
                else:
                    consistency_scores.append(0.5)  # Neutral score for empty concept sets
            
            return float(np.mean(consistency_scores)) if consistency_scores else 1.0
        except Exception:
            return 1.0
    
    def _assess_logical_flow(self, text: str) -> float:
        """Assess logical flow within text"""
        try:
            # Look for logical connectors and transitions
            logical_connectors = {
                'causal': ['because', 'since', 'due to', 'as a result', 'therefore', 'thus', 'consequently'],
                'additive': ['also', 'furthermore', 'moreover', 'in addition', 'additionally'],
                'contrastive': ['however', 'but', 'nevertheless', 'on the other hand', 'conversely'],
                'sequential': ['first', 'then', 'next', 'finally', 'subsequently', 'afterwards']
            }
            
            text_lower = text.lower()
            connector_counts = {}
            
            for category, connectors in logical_connectors.items():
                count = sum(1 for connector in connectors if connector in text_lower)
                connector_counts[category] = count
            
            total_connectors = sum(connector_counts.values())
            word_count = len(text.split())
            
            if word_count == 0:
                return 0.0
            
            # Good logical flow has appropriate density of connectors
            connector_density = total_connectors / (word_count / 20)  # per 20 words
            
            # Optimal density is around 0.5-1.5 connectors per 20 words
            if 0.5 <= connector_density <= 1.5:
                flow_score = 1.0
            elif connector_density < 0.5:
                flow_score = connector_density / 0.5  # Penalty for too few connectors
            else:
                flow_score = max(0.0, 1.0 - (connector_density - 1.5) / 2.0)  # Penalty for too many
            
            # Bonus for variety in connector types
            variety_bonus = len([c for c in connector_counts.values() if c > 0]) / len(logical_connectors)
            
            return min(1.0, flow_score + variety_bonus * 0.2)
        except Exception:
            return 0.5

# ✅ MIXED PRECISION: Enhanced standalone utility functions for direct usage
def fsu_mathematical_accuracy(generated: str, target: str) -> float:
    """Mathematical solution accuracy evaluation for Phase 2 reasoning with mixed precision"""
    computer = FSUMetricsComputer(use_mixed_precision=True)
    return computer._evaluate_mathematical_accuracy(generated, target)

def fsu_reasoning_step_quality(generated: str, target: str) -> float:
    """Step-by-step reasoning quality assessment for Phase 2 with mixed precision"""
    computer = FSUMetricsComputer(use_mixed_precision=True)
    return computer._evaluate_reasoning_steps(generated, target)

def fsu_logical_consistency(text: str) -> float:
    """Logical consistency verification for reasoning tasks with mixed precision"""
    computer = FSUMetricsComputer(use_mixed_precision=True)
    return computer._evaluate_logical_consistency(text)

def fsu_context_scaling_performance(field: FSEField, context_lengths: List[int]) -> Dict[str, float]:
    """Measure performance degradation across different context lengths with mixed precision"""
    computer = FSUMetricsComputer(field.device, use_mixed_precision=True)
    
    performance_metrics = {}
    
    # ✅ MIXED PRECISION: Convert to fp32 for accurate analysis
    field_fp32 = computer._ensure_fp32_for_metrics(field)
    base_coherence = computer.compute_fse_sequence_coherence_fp32(field_fp32)
    
    for length in context_lengths:
        if length <= field_fp32.shape[1]:
            # Extract subsequence
            subfield_data = field_fp32[:, :length, :]
            
            # Compute coherence for this length
            length_coherence = computer.compute_fse_sequence_coherence_fp32(subfield_data)
            
            # Performance ratio (higher is better)
            performance_ratio = length_coherence / (base_coherence + 1e-8)
            performance_metrics[f'context_{length}'] = float(performance_ratio)
    
    return performance_metrics

def compute_fsu_language_metrics(predictions: Dict[str, FSEField], targets: Dict[str, FSEField], 
                                device: str = "gpu", generated_text: str = None, 
                                target_text: str = None) -> Dict[str, float]:
    """
    ✅ MIXED PRECISION: Comprehensive FSU language metrics computation - main API function
    """
    computer = FSUMetricsComputer(device, use_mixed_precision=True)
    
    # Core metrics with mixed precision optimization
    metrics = computer.compute_all_metrics(
        predictions, targets, 
        generated_text=generated_text, 
        target_text=target_text
    )
    
    # Add Phase 2 specific metrics if text provided
    if generated_text and target_text:
        metrics['fsu_mathematical_accuracy'] = fsu_mathematical_accuracy(generated_text, target_text)
        metrics['fsu_reasoning_step_quality'] = fsu_reasoning_step_quality(generated_text, target_text)
        metrics['fsu_logical_consistency'] = fsu_logical_consistency(generated_text)
    
    # Add context scaling metrics if semantic field available
    semantic_field_keys = [key for key in predictions.keys() if 'semantic' in key and isinstance(predictions[key], FSEField)]
    if semantic_field_keys:
        for field_key in semantic_field_keys[:1]:  # Process first semantic field
            context_lengths = [512, 1024, 2048, 4096]
            scaling_metrics = fsu_context_scaling_performance(predictions[field_key], context_lengths)
            # Add field-specific prefix
            field_scaling_metrics = {f"{field_key}_{k}": v for k, v in scaling_metrics.items()}
            metrics.update(field_scaling_metrics)
    
    return metrics

# ✅ MIXED PRECISION: Backward compatibility functions for direct usage
def compute_fsu_metrics(predictions: Dict[str, FSEField], targets: Dict[str, FSEField], 
                       device: str = "gpu", **kwargs) -> Dict[str, float]:
    """Compute comprehensive FSU metrics - backward compatibility function with mixed precision"""
    computer = FSUMetricsComputer(device, use_mixed_precision=True)
    return computer.compute_all_metrics(predictions, targets, **kwargs)

# ✅ MIXED PRECISION: Individual metric functions for specific use cases
def fsu_perplexity(pred_logits: FSEField, target_chars: FSEField) -> float:
    """FSU language modeling perplexity with mixed precision optimization"""
    computer = FSUMetricsComputer(pred_logits.device, use_mixed_precision=True)
    metrics = computer.compute_language_modeling_metrics(pred_logits, target_chars)
    return metrics.get('fsu_perplexity', float('inf'))

def fsu_field_coherence(field: FSEField) -> float:
    """FSU semantic field coherence with mixed precision optimization"""
    computer = FSUMetricsComputer(field.device, use_mixed_precision=True)
    field_fp32 = computer._ensure_fp32_for_metrics(field)
    return computer.compute_fse_sequence_coherence_fp32(field_fp32)

def fsu_conversation_coherence(generated_text: str, target_text: str, context: List[str] = None) -> float:
    """FSU conversation coherence with enhanced analysis"""
    computer = FSUMetricsComputer(use_mixed_precision=True)
    metrics = computer.compute_conversation_metrics(generated_text, target_text, context)
    return metrics.get('fsu_conversation_coherence', 0.5)

def fsu_reasoning_accuracy(generated: str, target: str) -> float:
    """FSU reasoning accuracy with mixed precision optimization"""
    computer = FSUMetricsComputer(use_mixed_precision=True)
    if computer._contains_math(target):
        return computer._evaluate_mathematical_accuracy(generated, target)
    else:
        return computer._evaluate_logical_consistency(generated)

def fsu_response_quality(generated: str, target: str) -> float:
    """Overall FSU response quality with enhanced metrics"""
    computer = FSUMetricsComputer(use_mixed_precision=True)
    bleu = computer._compute_bleu_score(generated, target)
    fluency = computer._compute_text_fluency(generated)
    relevance = computer._compute_response_relevance(generated, target)
    diversity = computer._compute_response_diversity(generated)
    consistency = computer._compute_semantic_consistency(generated, target)
    
    # Weighted combination of quality metrics
    quality_score = (bleu * 0.25 + fluency * 0.25 + relevance * 0.2 + 
                    diversity * 0.15 + consistency * 0.15)
    
    return quality_score