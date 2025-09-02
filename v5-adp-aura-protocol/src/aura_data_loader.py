# file: aura_data_loader.py (Cleaned Version)
# Removed all GCP/GCS dependencies for local research archive.

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger("AURA_DATA_LOADER")

class AuraDataLoader:
    """[LOCAL ARCHIVE VERSION] Cleaned local-only AURA data loader."""
    def __init__(self, data_dir: str, batch_size: int):
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        if not os.path.exists(data_dir):
            logger.warning(f"⚠️ AURA data directory not found: {data_dir}")
