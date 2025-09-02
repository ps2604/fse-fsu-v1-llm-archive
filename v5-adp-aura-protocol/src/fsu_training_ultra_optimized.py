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
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import cupy as cp

# Import local FSU components
try:
    from adjoint_fsu_model import FSULanguageModel
    from adjoint_loss_functions import FSULossFunctions
    from adjoint_core_optimized import FSEField, FieldType
    from fsu_async_data_loader import FSUAsyncDataLoader
    from metrics_fsu import FSUMetrics
except ImportError:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FSU_TRAINING")

class FSECheckpointManager:
    """[LOCAL ARCHIVE VERSION] Cleaned local checkpoint manager."""
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.max_checkpoints = max_checkpoints
        self.local_dir = checkpoint_dir
        os.makedirs(self.local_dir, exist_ok=True)
        logger.info(f"✅ Local Checkpoint Manager initialized: '{self.local_dir}'")
            
    def save_checkpoint(self, model, optimizer, epoch: int, step: int, loss: float, args):
        try:
            checkpoint_data = {
                'epoch': epoch, 'step': step, 'loss': loss,
                'model_parameters': self._serialize_params(model.parameters),
                'optimizer_state': getattr(optimizer, 'state_dict', lambda: {})()
            }
            path = os.path.join(self.local_dir, f"checkpoint_e{epoch}_s{step}.fse")
            np.save(path, checkpoint_data, allow_pickle=True)
            logger.info(f"✅ Saved: {path}")
            return path
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")
            return None

    def _serialize_params(self, params: Dict) -> Dict:
        serialized = {}
        for k, v in params.items():
            if isinstance(v, dict): serialized[k] = self._serialize_params(v)
            elif hasattr(v, 'data'):
                d = v.data
                if hasattr(d, 'get'): d = d.get() # cupy -> numpy
                serialized[k] = {'data': d}
        return serialized
