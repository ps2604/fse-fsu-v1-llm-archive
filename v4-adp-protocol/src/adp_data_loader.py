# file: adp_data_loader.py (Cleaned Version)
# Removed all GCP/GCS dependencies for local research archive.

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger("ADP_DATA_LOADER")

class ADPDataLoader:
    """[LOCAL ARCHIVE VERSION] Cleaned local-only ADP data loader."""
    def __init__(self, data_dir: str, batch_size: int, rank: int = 0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.rank = rank
        
        if not os.path.exists(data_dir):
            logger.warning(f"⚠️ ADP data directory not found: {data_dir}")
