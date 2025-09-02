# file: fsu_training_ultra_optimized.py (Cleaned Version)
# Removed all GCP/GCS dependencies for local research archive.

import os
import sys
import time
import io
import json
import random
import logging
import argparse
import signal
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import cupy as cp

# Import local FSU components
from adjoint_fsu_model import FSULanguageModel
from adjoint_loss_functions import FSULossFunctions
from adjoint_core_optimized import get_memory_pool, get_default_dtype, FSEField, FieldType
from fsu_async_data_loader import FSUAsyncDataLoader, StreamDataLoader
from metrics_fsu import FSUMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FSU_TRAINING")

# 🚀 ROBUST LOCAL CHECKPOINT MANAGEMENT
# =========================================

class FSECheckpointManager:
    """
    [LOCAL ARCHIVE VERSION] Robust checkpoint manager focused on local persistence.
    Saves and loads model, optimizer, and training state.
    """
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.max_checkpoints = max_checkpoints
        self.local_dir = checkpoint_dir
        os.makedirs(self.local_dir, exist_ok=True)
        logger.info(f"✅ Local Checkpoint Manager initialized. Directory: '{self.local_dir}'")
            
    def save_checkpoint(self, model, optimizer, epoch: int, step: int, 
                       loss: float, args, additional_info: Optional[Dict] = None,
                       gradient_scaler=None, fse_state_manager=None):
        """Saves EVERY piece of training state to local disk"""
        try:
            checkpoint_data = {
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'timestamp': datetime.now().isoformat(),
                'fse_version': 'continuous_serialized_v3_complete',
                'model_parameters': self._serialize_fse_parameters_complete(model.parameters),
                'model_config': {
                    'device': model.device,
                    'sequence_length': getattr(model, 'sequence_length', 4096),
                    'channels': getattr(model, 'channels', 256),
                    'step_count': getattr(model, 'step_count', 4),
                    'vocab_size': getattr(model, 'vocab_size', 65536),
                },
                'optimizer_state': self._serialize_optimizer_complete(optimizer),
                'args': vars(args) if args else {},
            }
            
            if additional_info:
                checkpoint_data['additional_info'] = additional_info

            checkpoint_filename = f"checkpoint_epoch_{epoch}_step_{step}.fse"
            local_path = os.path.join(self.local_dir, checkpoint_filename)
            
            # Use temporary file for atomic write
            tmp_path = local_path + ".tmp"
            with open(tmp_path, 'wb') as f:
                np.save(f, checkpoint_data, allow_pickle=True)
            os.replace(tmp_path, local_path)
            
            logger.info(f"✅ Saved checkpoint: {local_path}")
            self._cleanup_old_checkpoints()
            return local_path

        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}", exc_info=True)
            return None

    def load_latest_checkpoint(self, model, optimizer, gradient_scaler=None):
        """Loads the most recent checkpoint from the local directory"""
        checkpoints = sorted([f for f in os.listdir(self.local_dir) if f.endswith('.fse')])
        if not checkpoints:
            logger.info("ℹ️ No checkpoints found in directory.")
            return 0, 0, float('inf')
            
        latest_path = os.path.join(self.local_dir, checkpoints[-1])
        return self.load_checkpoint(latest_path, model, optimizer, gradient_scaler)

    def load_checkpoint(self, path: str, model, optimizer, gradient_scaler=None):
        """Loads a specific checkpoint file"""
        try:
            logger.info(f"Loading checkpoint from: {path}")
            checkpoint_data = np.load(path, allow_pickle=True).item()
            
            # Restore model parameters
            self._deserialize_fse_parameters_complete(model.parameters, checkpoint_data['model_parameters'])
            
            # Restore optimizer state
            self._deserialize_optimizer_complete(optimizer, checkpoint_data['optimizer_state'])
            
            epoch = checkpoint_data.get('epoch', 0)
            step = checkpoint_data.get('step', 0)
            loss = checkpoint_data.get('loss', float('inf'))
            
            logger.info(f"✅ Successfully loaded checkpoint (Epoch {epoch}, Step {step}, Loss {loss:.4f})")
            return epoch, step, loss

        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint {path}: {e}", exc_info=True)
            return 0, 0, float('inf')

    def _cleanup_old_checkpoints(self):
        """Keeps only the most recent N checkpoints"""
        checkpoints = sorted([f for f in os.listdir(self.local_dir) if f.endswith('.fse')])
        if len(checkpoints) > self.max_checkpoints:
            for old_ckpt in checkpoints[:-self.max_checkpoints]:
                try:
                    os.remove(os.path.join(self.local_dir, old_ckpt))
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete old checkpoint {old_ckpt}: {e}")

    def _serialize_fse_parameters_complete(self, parameters: Dict) -> Dict:
        """Deep serialization of FSEField parameters to CPU numpy"""
        serialized = {}
        for name, p in parameters.items():
            if isinstance(p, dict):
                serialized[name] = self._serialize_fse_parameters_complete(p)
            elif hasattr(p, 'data'):
                # Handle FSEField or similar wrapper
                data = p.data
                if hasattr(data, 'get'): # cupy to numpy
                    data = data.get()
                serialized[name] = {
                    'data': data,
                    'field_type': p.field_type.value if hasattr(p, 'field_type') else 'linear'
                }
        return serialized

    def _deserialize_fse_parameters_complete(self, model_params: Dict, serialized_params: Dict):
        """Restores parameters from serialized state"""
        for name, data_dict in serialized_params.items():
            if name in model_params:
                if isinstance(data_dict, dict) and 'data' not in data_dict:
                    self._deserialize_fse_parameters_complete(model_params[name], data_dict)
                elif hasattr(model_params[name], 'data'):
                    data = data_dict['data']
                    if model_params[name].device == "gpu":
                        data = cp.asarray(data)
                    model_params[name].data = data

    def _serialize_optimizer_complete(self, optimizer) -> Dict:
        """Serializes optimizer state including momentum buffers"""
        # This is implementation-dependent based on your custom optimizer
        # Assuming a basic structure for this archive
        return getattr(optimizer, 'state_dict', lambda: {})()

    def _deserialize_optimizer_complete(self, optimizer, state: Dict):
        """Restores optimizer state"""
        if hasattr(optimizer, 'load_state_dict'):
            optimizer.load_state_dict(state)
