# file: aura_utils.py (Cleaned Version)
# Removed all GCP/GCS dependencies for local research archive.

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

logger = logging.getLogger("AURA_UTILS")

class AuraArchiveManager:
    """[LOCAL ARCHIVE VERSION] Cleaned local-only archive manager."""
    def __init__(self, archive_path: str):
        self.archive_path = archive_path
        if not os.path.exists(archive_path):
            logger.warning(f"⚠️ Archive path not found: {archive_path}")
            
    def load_metadata(self):
        # Local logic only
        pass
