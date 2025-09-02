# SPDX-License-Identifier: Apache-2.0
# ADP CORE IMPLEMENTATION – V1.1 - AURALITH DATA-FIELD PROTOCOL
# Native data ecosystem for Float-Native State Elements (FSE) architecture
# Implements continuous field data representation with physics-aware metadata

import os
import json
import yaml
import zarr
import numpy as np
import cupy as cp
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from enum import Enum
import gcsfs
from zarr.storage import GCSMap

try:
    from adjoint_core_optimized import FSEField, FieldType, get_default_dtype
    from adjoint_fsu_model import FSULanguageModel
except ImportError:
    # Fallback for development
    FSEField = Any
    FieldType = Any

logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO, format_string=None):
    """Setup logging configuration - call only from main scripts"""
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(level=level, format=format_string, force=True)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)

class ADPVersion(Enum):
    """ADP Protocol Version Management"""
    V1_1 = "ADP-v1.1"

class FieldChannelType(Enum):
    """Standard ADP field channel types for FSE processing"""
    # Core FSE channels
    CONTINUOUS_STATE = "continuous_state"
    FIELD_MAGNITUDE = "field_magnitude"
    FIELD_PHASE = "field_phase"
    FIELD_COHERENCE = "field_coherence"
    
    # Physics channels for adjoint solving
    GRADIENT_X = "gradient_x"
    GRADIENT_Y = "gradient_y" 
    GRADIENT_Z = "gradient_z"
    LAPLACIAN = "laplacian"
    DIVERGENCE = "divergence"
    CURL_X = "curl_x"
    CURL_Y = "curl_y"
    CURL_Z = "curl_z"
    
    # Semantic channels for FSU processing
    SEMANTIC_DENSITY = "semantic_density"
    LINGUISTIC_FLOW = "linguistic_flow"
    DISCOURSE_STRUCTURE = "discourse_structure"
    PRAGMATIC_CONTEXT = "pragmatic_context"
    
    # Environmental channels for FLUXA processing
    LIGHT_TRANSPORT = "light_transport"
    MATERIAL_PROPERTIES = "material_properties"
    SURFACE_NORMALS = "surface_normals"
    DEPTH_FIELD = "depth_field"
    
    # Meta channels
    UNCERTAINTY = "uncertainty"
    QUALITY_METRIC = "quality_metric"
    TEMPORAL_COHERENCE = "temporal_coherence"

@dataclass
class ADPMetadata:
    """ADP v1.1 metadata structure with physics-aware fields and explicit time support"""
    protocol_version: str = ADPVersion.V1_1.value
    source_file: Optional[str] = None
    sha256_field: Optional[str] = None
    coordinate_system: str = "euclidean"
    voxel_size_m: Optional[List[float]] = None
    
    # Explicit time dimension support
    dimension_meaning: List[str] = None  # e.g., ["time", "x", "y", "channels"] or ["x", "y", "channels"]
    timestep_quantity: Optional[str] = None  # "milliseconds", "frames", "tokens", "days", etc.
    timestep_value: Optional[float] = None   # actual timestep value
    temporal_coherence: bool = True          # whether time evolution is coherent
    
    # Legacy timestep support for backward compatibility
    timestep_ms: Optional[float] = None
    
    units: Dict[str, Dict[str, Any]] = None
    licence: str = "Auralith Proprietary - Internal Use Only"
    generated_by_compiler: bool = True
    field_type: str = FieldType.CONTINUOUS.name if FieldType else "CONTINUOUS"
    evolution_rate: float = 0.05
    channels: Dict[str, Dict[str, Any]] = None
    physics_constants: Dict[str, float] = None
    boundary_conditions: Dict[str, Any] = None
    
    # Channel index mapping for physics weighting
    channel_index_map: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        if self.units is None:
            self.units = self._default_units()
        if self.channels is None:
            self.channels = self._default_channels()
        if self.physics_constants is None:
            self.physics_constants = self._default_physics_constants()
        if self.boundary_conditions is None:
            self.boundary_conditions = self._default_boundary_conditions()
    
    def _default_units(self) -> Dict[str, Dict[str, Any]]:
        """Default FSE-aware unit definitions"""
        return {
            "spatial": {"unit": "meter", "scale": 1.0, "range": [-1.0, 1.0]},
            "temporal": {"unit": "second", "scale": 1.0, "range": [0.0, float('inf')]},
            "field_magnitude": {"unit": "normalized", "scale": 1.0, "range": [-1.0, 1.0]},
            "field_phase": {"unit": "radians", "scale": 1.0, "range": [0.0, 2*np.pi]},
            "semantic_density": {"unit": "normalized", "scale": 1.0, "range": [0.0, 1.0]}
        }
    
    def _default_channels(self) -> Dict[str, Dict[str, Any]]:
        """Default channel configuration for FSE fields"""
        return {
            FieldChannelType.CONTINUOUS_STATE.value: {
                "dtype": "float32", "physics_type": "continuous_field",
                "evolution_rate": 0.05, "boundary_type": "periodic"
            },
            FieldChannelType.FIELD_MAGNITUDE.value: {
                "dtype": "float32", "physics_type": "scalar_field", 
                "evolution_rate": 0.02, "boundary_type": "neumann"
            },
            FieldChannelType.SEMANTIC_DENSITY.value: {
                "dtype": "float32", "physics_type": "density_field",
                "evolution_rate": 0.1, "boundary_type": "dirichlet"
            }
        }
    
    def _default_physics_constants(self) -> Dict[str, float]:
        """Physics constants for FSE field evolution"""
        try:
            from fse_physics_defaults import DEFAULT_PHYSICS_CONSTANTS
            return DEFAULT_PHYSICS_CONSTANTS.copy()
        except ImportError:
            # Fallback to hardcoded values
            return {
                "field_coupling_strength": 0.1,
                "diffusion_coefficient": 0.01,
                "evolution_damping": 0.95,
                "boundary_reflection": 0.8,
                "field_coherence_threshold": 0.1
            }
    
    def _default_boundary_conditions(self) -> Dict[str, Any]:
        """Default boundary conditions for FSE adjoint solving"""
        try:
            from fse_physics_defaults import DEFAULT_BOUNDARY_CONDITIONS
            return DEFAULT_BOUNDARY_CONDITIONS.copy()
        except ImportError:
            # Fallback to hardcoded values
            return {
                "type": "mixed",
                "dirichlet_faces": [],
                "neumann_faces": [],
                "periodic_faces": ["x", "y"],
                "reflection_coefficient": 0.8
            }

@dataclass 
class ADPDatapoint:
    """Individual ADP datapoint specification"""
    id: str
    path: str
    sha256: str
    split: str = "train"
    curriculum_stage: int = 1
    field_shape: Optional[Tuple[int, ...]] = None
    metadata: Optional[ADPMetadata] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = ADPMetadata()

@dataclass
class ADPManifest:
    """ADP v1.1 manifest structure"""
    protocol_version: str = ADPVersion.V1_1.value
    dataset_name: str = "Auralith FSE Dataset"
    checksum_algorithm: str = "sha256"
    created_at: str = ""
    datapoints: List[ADPDatapoint] = None
    
    def __post_init__(self):
        if self.created_at == "":
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if self.datapoints is None:
            self.datapoints = []

class ADPWriter:
    """Production-quality ADP format writer with FSE field optimization"""
    
    def __init__(self, 
                 root_path: Union[str, Path],
                 default_chunk_size: Tuple[int, ...] = (64, 128, 16),
                 compression: str = "blosc",
                 device: str = "gpu",
                 consolidate_metadata: bool = True,
                 minimal_meta: bool = False):
        self.root_path = Path(root_path)
        self.default_chunk_size = default_chunk_size
        self.compression = compression
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.consolidate_metadata = consolidate_metadata
        self.minimal_meta = minimal_meta
        
        # Create root directory structure
        self.root_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ ADPWriter initialized: {self.root_path}, compression={compression}, minimal_meta={minimal_meta}")
    
    def write_datapoint(self, 
                       datapoint_id: str,
                       field_data: np.ndarray,
                       metadata: ADPMetadata,
                       labels: Optional[Dict[str, np.ndarray]] = None,
                       pyramid_levels: int = 2) -> ADPDatapoint:
        """
        Write a complete ADP datapoint with field data, metadata, and optional labels.
        Handles explicit time dimension as first axis.
        """
        datapoint_path = self.root_path / datapoint_id
        datapoint_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure time is always the first dimension
        processed_field_data, processed_metadata = self._ensure_time_dimension(field_data, metadata)
        
        # Write main field data with optimized chunking for FSE processing
        field_zarr_path = datapoint_path / "field.zarr"
        self._write_field_zarr(processed_field_data, field_zarr_path, processed_metadata)
        
        # Generate multi-scale pyramid for hierarchical FSE processing
        if pyramid_levels > 0:
            self._write_pyramid(processed_field_data, datapoint_path / "pyramid", pyramid_levels, processed_metadata)
        
        # Write labels if provided
        if labels:
            self._write_labels(labels, datapoint_path / "labels")
        
        # Write metadata with full ADP v1.1 specification
        self._write_metadata(processed_metadata, datapoint_path / "meta.yaml")
        
        # Calculate SHA256 of the complete datapoint (handle minimal_meta runtime files)
        datapoint_sha256 = self._calculate_datapoint_hash(datapoint_path, exclude_runtime=self.minimal_meta)
        
        return ADPDatapoint(
            id=datapoint_id,
            path=f"{datapoint_id}/",
            sha256=datapoint_sha256,
            field_shape=processed_field_data.shape,
            metadata=processed_metadata
        )
    
    def _ensure_time_dimension(self, field_data: np.ndarray, metadata: ADPMetadata) -> Tuple[np.ndarray, ADPMetadata]:
        """
        Ensure time is always the first dimension for consistency across FSE domains.
        If time is not present, add singleton time dimension.
        """
        # Initialize dimension_meaning if not set
        if metadata.dimension_meaning is None:
            # Infer dimensions based on field shape
            if len(field_data.shape) == 2:  # (sequence, channels) - language
                metadata.dimension_meaning = ["x", "channels"]
            elif len(field_data.shape) == 3:  # (height, width, channels) - vision
                metadata.dimension_meaning = ["y", "x", "channels"]
            elif len(field_data.shape) == 4:  # (depth, height, width, channels) - 3D
                metadata.dimension_meaning = ["z", "y", "x", "channels"]
            else:
                # Generic fallback
                spatial_dims = [f"dim_{i}" for i in range(len(field_data.shape) - 1)]
                metadata.dimension_meaning = spatial_dims + ["channels"]
        
        # Check if time is already first dimension
        if len(metadata.dimension_meaning) > 0 and metadata.dimension_meaning[0] == "time":
            # Time already present
            return field_data, metadata
        
        # Determine if we should add time dimension based on domain/metadata
        should_add_time = self._should_add_time_dimension(field_data, metadata)
        
        if should_add_time:
            # Add singleton time dimension
            field_data_with_time = field_data[None, ...]  # Add time axis at front
            
            # Update dimension meaning
            new_dimension_meaning = ["time"] + metadata.dimension_meaning
            metadata.dimension_meaning = new_dimension_meaning
            
            # Set default time metadata if not present
            if metadata.timestep_quantity is None:
                metadata.timestep_quantity = self._infer_time_quantity(metadata)
            if metadata.timestep_value is None:
                metadata.timestep_value = 1.0  # Single timestep
            
            return field_data_with_time, metadata
        else:
            # Keep as static (no time dimension)
            return field_data, metadata
    
    def _should_add_time_dimension(self, field_data: np.ndarray, metadata: ADPMetadata) -> bool:
        """Determine if time dimension should be added based on domain and data characteristics"""
        # Check coordinate system for clues
        if metadata.coordinate_system == "semantic_manifold_1d":
            return True  # Language domain - sequence is temporal
        
        # Check if timestep information is provided
        if metadata.timestep_quantity is not None or metadata.timestep_value is not None:
            return True
        
        # For very long sequences, assume temporal nature
        if len(field_data.shape) >= 2 and field_data.shape[0] > 1000:
            return True
        
        # Default: static data (images, single volumes, etc.)
        return False
    
    def _infer_time_quantity(self, metadata: ADPMetadata) -> str:
        """Infer time quantity based on coordinate system and other metadata"""
        if metadata.coordinate_system == "semantic_manifold_1d":
            return "tokens"
        elif "seismic" in str(metadata.source_file).lower():
            return "milliseconds"
        elif "video" in str(metadata.source_file).lower() or "frame" in str(metadata.source_file).lower():
            return "frames"
        else:
            return "steps"  # Generic time steps
    
    def _write_field_zarr(self, field_data: np.ndarray, zarr_path: Path, metadata: ADPMetadata):
        """Write field data with FSE-optimized chunking and compression"""
        # Ensure data is in the correct format for FSE processing
        if field_data.dtype != np.float32:
            field_data = field_data.astype(np.float32)
        
        # Calculate optimal chunk size based on field dimensions and FSE processing patterns
        chunk_size = self._calculate_optimal_chunks(field_data.shape)
        
        # Create Zarr store with high-performance codec
        store = zarr.DirectoryStore(str(zarr_path))
        
        # Write with compression optimized for continuous fields
        z = zarr.open(store, mode='w', 
                     shape=field_data.shape, 
                     chunks=chunk_size,
                     dtype=field_data.dtype,
                     compression=self.compression,
                     compression_opts={'cname': 'zstd', 'clevel': 3, 'shuffle': 2})
        
        z[:] = field_data
        
        # Store FSE-specific attributes (always include expected_channels)
        core_attrs = {
            'field_type': metadata.field_type,
            'evolution_rate': metadata.evolution_rate,
            'coordinate_system': metadata.coordinate_system,
            'adp_version': metadata.protocol_version,
            'expected_channels': field_data.shape[-1] if len(field_data.shape) > 1 else 1  # Always write
        }
        
        if not self.minimal_meta:
            core_attrs['physics_constants'] = metadata.physics_constants
        
        z.attrs.update(core_attrs)
        
        # Consolidate metadata for faster loading
        if self.consolidate_metadata:
            zarr.consolidate_metadata(store)
        
        logger.debug(f"✅ Written field zarr: {zarr_path}, shape={field_data.shape}, chunks={chunk_size}")
    
    def _write_pyramid(self, field_data: np.ndarray, pyramid_path: Path, levels: int, metadata: ADPMetadata):
        """Generate multi-scale pyramid for hierarchical FSE processing - Handle time dimension properly"""
        pyramid_path.mkdir(parents=True, exist_ok=True)
        
        current_data = field_data
        for level in range(1, levels + 1):
            # FIXED: Handle time dimension properly in pyramid pooling
            has_time = (hasattr(metadata, 'dimension_meaning') and 
                       metadata.dimension_meaning and 
                       metadata.dimension_meaning[0] == "time")
            
            if has_time and len(current_data.shape) > 3:
                # Time-aware pooling: pool only spatial dimensions, preserve time
                pooled = self._average_pool_2x_time_aware(current_data)
            elif len(current_data.shape) >= 3:  # Spatial dimensions present
                pooled = self._average_pool_2x(current_data)
            else:
                # For 1D/2D data, use different pooling strategy
                pooled = current_data[::2] if len(current_data.shape) == 1 else current_data[::2, ::2]
            
            level_path = pyramid_path / f"level_{level}.zarr"
            store = zarr.DirectoryStore(str(level_path))
            
            chunk_size = self._calculate_optimal_chunks(pooled.shape)
            z = zarr.open(store, mode='w',
                         shape=pooled.shape,
                         chunks=chunk_size, 
                         dtype=pooled.dtype,
                         compression=self.compression)
            z[:] = pooled
            
            current_data = pooled
            logger.debug(f"✅ Written pyramid level {level}: shape={pooled.shape}")
    
    def _write_labels(self, labels: Dict[str, np.ndarray], labels_path: Path):
        """Write semantic labels and masks for supervised FSE training"""
        labels_path.mkdir(parents=True, exist_ok=True)
        
        for label_name, label_data in labels.items():
            label_zarr_path = labels_path / f"{label_name}.zarr"
            store = zarr.DirectoryStore(str(label_zarr_path))
            
            chunk_size = self._calculate_optimal_chunks(label_data.shape)
            z = zarr.open(store, mode='w',
                         shape=label_data.shape,
                         chunks=chunk_size,
                         dtype=label_data.dtype,
                         compression=self.compression)
            z[:] = label_data
            
            logger.debug(f"✅ Written label '{label_name}': shape={label_data.shape}")
    
    def _write_metadata(self, metadata: ADPMetadata, meta_path: Path):
        """Write ADP v1.1 compliant metadata with physics-aware fields"""
        if self.minimal_meta:
            # Write only essential metadata for performance
            minimal_metadata = {
                'protocol_version': metadata.protocol_version,
                'field_type': metadata.field_type,
                'evolution_rate': metadata.evolution_rate,
                'coordinate_system': metadata.coordinate_system,
                'channels': metadata.channels,
                'generated_by_compiler': metadata.generated_by_compiler
            }
            metadata_dict = minimal_metadata
        else:
            metadata_dict = asdict(metadata)
            
            # Add runtime metadata
            metadata_dict.update({
                'created_at': datetime.utcnow().isoformat() + "Z",
                'compiler_version': "FSE-Data-Compiler-v1.1",
                'fse_architecture_version': "FSE-v1.0"
            })
        
        with open(meta_path, 'w') as f:
            yaml.dump(metadata_dict, f, default_flow_style=False, sort_keys=False)
        
        # Write runtime sidecar if not minimal mode
        if not self.minimal_meta:
            runtime_path = meta_path.parent / "meta_runtime.yml"
            runtime_data = {
                'last_accessed': datetime.utcnow().isoformat() + "Z",
                'access_count': 1,
                'performance_metrics': {
                    'avg_load_time_ms': 0.0,
                    'cache_hits': 0
                }
            }
            with open(runtime_path, 'w') as f:
                yaml.dump(runtime_data, f, default_flow_style=False)
        
        logger.debug(f"✅ Written metadata: {meta_path}")
    
    def _calculate_optimal_chunks(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate optimal chunk sizes for FSE field processing - Use default_chunk_size as fallback"""
        # Use default_chunk_size as fallback
        if hasattr(self, 'default_chunk_size') and len(self.default_chunk_size) == len(shape):
            base_chunks = self.default_chunk_size
        else:
            base_chunks = None
        
        if len(shape) == 1:  # 1D sequence data (FSU)
            return (min(1024, shape[0]),) if not base_chunks else (min(base_chunks[0], shape[0]),)
        elif len(shape) == 2:  # 2D field data
            if base_chunks:
                return (min(base_chunks[0], shape[0]), min(base_chunks[1], shape[1]))
            return (min(128, shape[0]), min(128, shape[1]))
        elif len(shape) == 3:  # 3D field data  
            if base_chunks:
                return (min(base_chunks[0], shape[0]), min(base_chunks[1], shape[1]), min(base_chunks[2], shape[2]))
            return (min(64, shape[0]), min(128, shape[1]), min(16, shape[2]))
        else:  # Higher dimensional
            chunks = []
            for i, dim in enumerate(shape):
                if base_chunks and i < len(base_chunks):
                    chunks.append(min(base_chunks[i], dim))
                elif i < 2:  # Spatial dimensions
                    chunks.append(min(64, dim))
                else:  # Feature/channel dimensions
                    chunks.append(min(16, dim))
            return tuple(chunks)
    
    def _average_pool_2x(self, data: np.ndarray) -> np.ndarray:
        """2x average pooling preserving field properties - Handle odd dimensions"""
        if len(data.shape) == 3:  # (H, W, C)
            h, w, c = data.shape
            # Handle odd dimensions with padding
            if h % 2 == 1:
                # Reflect pad the last row
                data = np.concatenate([data, data[-1:]], axis=0)
                h += 1
            if w % 2 == 1:
                # Reflect pad the last column
                data = np.concatenate([data, data[:, -1:]], axis=1)
                w += 1
                
            new_h, new_w = h // 2, w // 2
            reshaped = data[:new_h*2, :new_w*2, :].reshape(new_h, 2, new_w, 2, c)
            return reshaped.mean(axis=(1, 3))
        elif len(data.shape) == 4:  # (B, H, W, C)
            b, h, w, c = data.shape
            # Handle odd dimensions with padding
            if h % 2 == 1:
                data = np.concatenate([data, data[:, -1:]], axis=1)
                h += 1
            if w % 2 == 1:
                data = np.concatenate([data, data[:, :, -1:]], axis=2)
                w += 1
                
            new_h, new_w = h // 2, w // 2
            reshaped = data[:, :new_h*2, :new_w*2, :].reshape(b, new_h, 2, new_w, 2, c)
            return reshaped.mean(axis=(2, 4))
        else:
            raise ValueError(f"Unsupported pooling shape: {data.shape}")
    
    def _average_pool_2x_time_aware(self, data: np.ndarray) -> np.ndarray:
        """Time-aware 2x pooling: preserve time dimension, pool only spatial dims"""
        if len(data.shape) == 4:  # (T, H, W, C)
            t, h, w, c = data.shape
            # Handle odd spatial dimensions with padding
            if h % 2 == 1:
                data = np.concatenate([data, data[:, -1:]], axis=1)
                h += 1
            if w % 2 == 1:
                data = np.concatenate([data, data[:, :, -1:]], axis=2)
                w += 1
                
            new_h, new_w = h // 2, w // 2
            reshaped = data[:, :new_h*2, :new_w*2, :].reshape(t, new_h, 2, new_w, 2, c)
            return reshaped.mean(axis=(2, 4))
        elif len(data.shape) == 5:  # (T, D, H, W, C)
            t, d, h, w, c = data.shape
            # Handle odd spatial dimensions with padding
            if d % 2 == 1:
                data = np.concatenate([data, data[:, -1:]], axis=1)
                d += 1
            if h % 2 == 1:
                data = np.concatenate([data, data[:, :, -1:]], axis=2)
                h += 1
            if w % 2 == 1:
                data = np.concatenate([data, data[:, :, :, -1:]], axis=3)
                w += 1
                
            new_d, new_h, new_w = d // 2, h // 2, w // 2
            reshaped = data[:, :new_d*2, :new_h*2, :new_w*2, :].reshape(t, new_d, 2, new_h, 2, new_w, 2, c)
            return reshaped.mean(axis=(2, 4, 6))
        else:
            # Fallback to regular pooling for unexpected shapes
            return self._average_pool_2x(data)
    
    def _calculate_datapoint_hash(self, datapoint_path: Path, exclude_runtime: bool = False) -> str:
        """Calculate SHA256 hash of complete datapoint - Handle minimal_meta runtime files"""
        hasher = hashlib.sha256()
        
        # Hash all files in datapoint directory in deterministic order
        for file_path in sorted(datapoint_path.rglob('*')):
            if file_path.is_file():
                # Skip runtime file if exclude_runtime is True (for minimal_meta consistency)
                if exclude_runtime and file_path.name == "meta_runtime.yml":
                    continue
                    
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        hasher.update(chunk)
        
        return hasher.hexdigest()

class ADPReader:
    """
    [GCS-NATIVE V2.1 - COMPLETE] Production-quality ADP format reader that can read
    directly from either a local path or a GCS path (gs://...).
    - Restores all functionality from the original local-only reader.
    """
    
    def __init__(self, root_path: Union[str, Path], device: str = "gpu"):
        self.root_path_str = str(root_path)
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.is_gcs = False # Research Archive: Local-only
        self.gcs_fs = None

        if device == "gpu" and cp.cuda.is_available():
            self._setup_multi_gpu_memory_pools()
        
        self.manifest = self._load_manifest()
        
        logger.info(f"✅ ADPReader initialized for local path: {self.root_path_str}")
        logger.info(f"   Found {len(self.manifest.datapoints)} datapoints.")

    def _setup_multi_gpu_memory_pools(self):
        """Setup memory pools for all available GPUs."""
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            for device_id in range(device_count):
                with cp.cuda.Device(device_id):
                    free_bytes, _ = cp.cuda.Device(device_id).mem_info
                    max_safe_bytes = int(free_bytes * 0.9)
                    min_pool_size = min(30 * 1024**3, max_safe_bytes)
                    
                    pool = cp.get_default_memory_pool()
                    if pool.total_bytes() < min_pool_size:
                        pool.set_limit(size=min_pool_size)
        except Exception as e:
            logger.warning(f"⚠️ Could not configure multi-GPU memory pools: {e}")

    def _load_manifest(self) -> ADPManifest:
        """Load and validate ADP manifest from local or GCS."""
        manifest_path = os.path.join(self.root_path_str, "manifest.yml")
        try:
            if self.is_gcs:
                with self.gcs_fs.open(manifest_path, 'r') as f:
                    manifest_data = yaml.safe_load(f)
            else:
                if not os.path.exists(manifest_path):
                    raise FileNotFoundError(f"ADP manifest not found: {manifest_path}")
                with open(manifest_path, 'r') as f:
                    manifest_data = yaml.safe_load(f)
            
            datapoints = [ADPDatapoint(**dp) for dp in manifest_data.get('datapoints', [])]
            return ADPManifest(**{k: v for k, v in manifest_data.items() if k != 'datapoints'}, datapoints=datapoints)
        except Exception as e:
            logger.error(f"Failed to load manifest from {manifest_path}: {e}")
            raise

    def load_datapoint(self, datapoint_id: str, load_pyramid: bool = False, 
                       pyramid_level: int = 0, return_backend: str = 'numpy') -> Tuple[Any, ADPMetadata, Optional[Dict[str, Any]]]:
        """Load a complete ADP datapoint with pyramid support."""
        datapoint_info = next((dp for dp in self.manifest.datapoints if dp.id == datapoint_id), None)
        if not datapoint_info:
            raise KeyError(f"Datapoint '{datapoint_id}' not found in manifest.")
            
        datapoint_path = os.path.join(self.root_path_str, datapoint_info.path)
        
        metadata = self._load_metadata(os.path.join(datapoint_path, "meta.yaml"))
        
        if load_pyramid and pyramid_level > 0:
            field_data_path = os.path.join(datapoint_path, "pyramid", f"level_{pyramid_level}.zarr")
        else:
            field_data_path = os.path.join(datapoint_path, "field.zarr")
        
        field_data = self._load_field_zarr(field_data_path, return_backend)
        
        labels_path = os.path.join(datapoint_path, "labels")
        labels = self._load_labels(labels_path, return_backend) if self._path_exists(labels_path) else None
        
        return field_data, metadata, labels

    def _path_exists(self, path: str) -> bool:
        if self.is_gcs:
            return self.gcs_fs.exists(path)
        else:
            return os.path.exists(path)

    def _load_metadata(self, meta_path: str) -> ADPMetadata:
        if self.is_gcs:
            with self.gcs_fs.open(meta_path, 'r') as f:
                meta_dict = yaml.safe_load(f)
        else:
            with open(meta_path, 'r') as f:
                meta_dict = yaml.safe_load(f)
        return ADPMetadata(**{k: v for k, v in meta_dict.items() if k in ADPMetadata.__dataclass_fields__})

    def _load_field_zarr(self, zarr_path: str, target_backend_str: str = 'numpy') -> Any:
        if self.is_gcs:
            store = GCSMap(zarr_path, gcs=self.gcs_fs, check=True)
        else:
            store = zarr.DirectoryStore(zarr_path)
            
        z = zarr.open(store, mode='r')
        field_data = np.array(z[:])
        
        target_backend = cp if target_backend_str == 'cupy' else np
        if target_backend == cp:
            field_data = cp.asarray(field_data)
        return field_data

    def _load_labels(self, labels_path: str, target_backend_str: str = 'numpy') -> Dict[str, Any]:
        labels = {}
        if self.is_gcs:
            label_files = [f for f in self.gcs_fs.ls(labels_path) if f.endswith('.zarr')]
        else:
            label_files = [str(p) for p in Path(labels_path).glob("*.zarr")]
        
        for label_file in label_files:
            label_name = Path(label_file).stem
            labels[label_name] = self._load_field_zarr(label_file, target_backend_str)
        return labels

    def get_datapoint_iterator(self, split: str = "train", batch_size: int = 1):
        filtered_datapoints = [dp for dp in self.manifest.datapoints if dp.split == split]
        for i in range(0, len(filtered_datapoints), batch_size):
            batch_datapoints = filtered_datapoints[i:i+batch_size]
            batch_data = []
            for dp in batch_datapoints:
                field_data, metadata, labels = self.load_datapoint(dp.id, return_backend='auto')
                batch_data.append((field_data, metadata, labels))
            yield batch_data


class ADPValidator:
    """ADP format validation and integrity checking"""
    
    def __init__(self, root_path: Union[str, Path]):
        self.root_path = Path(root_path)
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Comprehensive ADP dataset validation"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'datapoint_count': 0,
            'total_size_bytes': 0,
            'protocol_version': None
        }
        
        try:
            # Validate manifest
            manifest_validation = self._validate_manifest()
            validation_results.update(manifest_validation)
            
            if not manifest_validation['valid']:
                validation_results['valid'] = False
                return validation_results
            
            # Load manifest for datapoint validation
            reader = ADPReader(self.root_path)
            manifest = reader.manifest
            validation_results['protocol_version'] = manifest.protocol_version
            
            # Validate each datapoint
            for datapoint in manifest.datapoints:
                dp_validation = self._validate_datapoint(datapoint)
                if not dp_validation['valid']:
                    validation_results['valid'] = False
                    validation_results['errors'].extend(dp_validation['errors'])
                
                validation_results['warnings'].extend(dp_validation['warnings'])
                validation_results['total_size_bytes'] += dp_validation.get('size_bytes', 0)
            
            validation_results['datapoint_count'] = len(manifest.datapoints)
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation failed: {str(e)}")
        
        return validation_results
    
    def _validate_manifest(self) -> Dict[str, Any]:
        """Validate ADP manifest structure and content"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        manifest_path = self.root_path / "manifest.yml"
        if not manifest_path.exists():
            result['valid'] = False
            result['errors'].append("manifest.yml not found")
            return result
        
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = yaml.safe_load(f)
            
            # Check required fields
            required_fields = ['protocol_version', 'dataset_name', 'checksum_algorithm', 'datapoints']
            for field in required_fields:
                if field not in manifest_data:
                    result['errors'].append(f"Missing required field in manifest: {field}")
                    result['valid'] = False
            
            # Validate protocol version
            protocol_version = manifest_data.get('protocol_version')
            if protocol_version not in [v.value for v in ADPVersion]:
                result['warnings'].append(f"Unknown protocol version: {protocol_version}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Failed to parse manifest: {str(e)}")
        
        return result
    
    def _validate_datapoint(self, datapoint: ADPDatapoint) -> Dict[str, Any]:
        """Validate individual ADP datapoint"""
        result = {'valid': True, 'errors': [], 'warnings': [], 'size_bytes': 0}
        
        datapoint_path = self.root_path / datapoint.id
        if not datapoint_path.exists():
            result['valid'] = False
            result['errors'].append(f"Datapoint directory not found: {datapoint.id}")
            return result
        
        # Check required files
        required_files = ['field.zarr', 'meta.yaml']
        for req_file in required_files:
            file_path = datapoint_path / req_file
            if not file_path.exists():
                result['valid'] = False
                result['errors'].append(f"Required file missing: {datapoint.id}/{req_file}")
        
        # Validate field.zarr
        field_zarr_path = datapoint_path / "field.zarr"
        if field_zarr_path.exists():
            zarr_validation = self._validate_zarr(field_zarr_path)
            if not zarr_validation['valid']:
                result['valid'] = False
                result['errors'].extend(zarr_validation['errors'])
            result['size_bytes'] += zarr_validation.get('size_bytes', 0)
        
        # Validate metadata
        meta_path = datapoint_path / "meta.yaml"
        if meta_path.exists():
            meta_validation = self._validate_metadata(meta_path)
            if not meta_validation['valid']:
                result['valid'] = False
                result['errors'].extend(meta_validation['errors'])
        
        # Validate SHA256 if provided
        if datapoint.sha256:
            calculated_hash = self._calculate_datapoint_hash(datapoint_path)
            if calculated_hash != datapoint.sha256:
                result['valid'] = False
                result['errors'].append(f"SHA256 mismatch for datapoint {datapoint.id}")
        
        return result
    
    def _validate_zarr(self, zarr_path: Path) -> Dict[str, Any]:
        """Validate Zarr format and content"""
        result = {'valid': True, 'errors': [], 'size_bytes': 0}
        
        try:
            store = zarr.DirectoryStore(str(zarr_path))
            z = zarr.open(store, mode='r')
            
            # Check data integrity
            if z.shape is None or len(z.shape) == 0:
                result['valid'] = False
                result['errors'].append(f"Invalid Zarr shape: {zarr_path}")
            
            # Calculate size
            result['size_bytes'] = z.nbytes
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Failed to validate Zarr {zarr_path}: {str(e)}")
        
        return result
    
    def _validate_metadata(self, meta_path: Path) -> Dict[str, Any]:
        """Validate metadata YAML structure"""
        result = {'valid': True, 'errors': []}
        
        try:
            with open(meta_path, 'r') as f:
                meta_data = yaml.safe_load(f)
            
            # Check protocol version
            if 'protocol_version' not in meta_data:
                result['errors'].append(f"Missing protocol_version in {meta_path}")
                result['valid'] = False
            
            # Check required FSE fields
            fse_required = ['field_type', 'evolution_rate']
            for field in fse_required:
                if field not in meta_data:
                    result['errors'].append(f"Missing FSE field '{field}' in {meta_path}")
                    result['valid'] = False
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Failed to parse metadata {meta_path}: {str(e)}")
        
        return result
    
    def _calculate_datapoint_hash(self, datapoint_path: Path) -> str:
        """Calculate SHA256 hash for integrity verification"""
        hasher = hashlib.sha256()
        
        for file_path in sorted(datapoint_path.rglob('*')):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        hasher.update(chunk)
        
        return hasher.hexdigest()

# Utility functions for ADP manipulation
def create_adp_dataset(root_path: Union[str, Path], 
                       dataset_name: str = "Auralith FSE Dataset",
                       description: str = "") -> ADPWriter:
    """Initialize a new ADP dataset with proper directory structure"""
    root_path = Path(root_path)
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Create initial manifest
    manifest = ADPManifest(dataset_name=dataset_name)
    manifest_path = root_path / "manifest.yml"
    
    with open(manifest_path, 'w') as f:
        yaml.dump(asdict(manifest), f, default_flow_style=False)
    
    logger.info(f"✅ Created ADP dataset: {root_path}")
    return ADPWriter(root_path)

# In adp_core.py

def update_manifest(root_path: Union[str, Path], datapoint: ADPDatapoint):
    """Update ADP manifest with new datapoint"""
    manifest_path = Path(root_path) / "manifest.yml"
    
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest_data = yaml.safe_load(f)
    else:
        manifest_data = asdict(ADPManifest())
    
    # Add or update datapoint
    datapoints = manifest_data.get('datapoints', [])
    
    # Convert dataclass to dictionary
    datapoint_dict = asdict(datapoint)
    
    # ======================= THIS IS THE FIX =======================
    # Check if field_shape exists and is a tuple, then convert to a list
    if 'field_shape' in datapoint_dict and isinstance(datapoint_dict['field_shape'], tuple):
        datapoint_dict['field_shape'] = list(datapoint_dict['field_shape'])
    # ===============================================================
    
    # Remove existing datapoint with same ID
    datapoints = [dp for dp in datapoints if dp.get('id') != datapoint.id]
    datapoints.append(datapoint_dict)
    
    manifest_data['datapoints'] = datapoints
    manifest_data['updated_at'] = datetime.utcnow().isoformat() + "Z"
    
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest_data, f, default_flow_style=False)
    
    logger.debug(f"✅ Updated manifest with datapoint: {datapoint.id}")