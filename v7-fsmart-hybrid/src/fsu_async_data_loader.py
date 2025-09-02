# file: fsu_async_data_loader.py (Cleaned Version)
# Removed all GCP/GCS dependencies for local research archive.

import os
import sys
import time
import json
import random
import logging
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import cupy as cp

logger = logging.getLogger("FSU_DATA_LOADER")

class FSUAsyncDataLoader:
    """[LOCAL ARCHIVE VERSION] Cleaned local-only data loader."""
    def __init__(self, data_dir: str, batch_size: int, sequence_length: int, device: str = "gpu"):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device
        self.backend = cp if device == "gpu" else np
        
        # Original GCS logic removed for local archive
        if not os.path.exists(data_dir):
            logger.warning(f"⚠️ Data directory not found: {data_dir}")
            
    def _load_local_file(self, file_path: str):
        with open(file_path, 'r') as f:
            return json.load(f)

class StreamDataLoader:
    """[LOCAL ARCHIVE VERSION] Cleaned local-only stream loader."""
    def __init__(self, stream_file_path: str, batch_size: int, sequence_length: int, stride: int = 2048):
        self.stream_file_path = stream_file_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Local-only check
        if not os.path.exists(stream_file_path):
            logger.warning(f"⚠️ Stream file not found: {stream_file_path}")
