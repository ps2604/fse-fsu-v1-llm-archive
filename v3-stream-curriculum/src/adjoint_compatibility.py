# adjoint_compatibility.py
"""Compatibility layer between old and new adjoint implementations"""

# Re-export adjoint implementations with old names for compatibility
from adjoint_core_optimized import *
from adjoint_components import *
from adjoint_fsu_model import FSULanguageModel
from adjoint_loss_functions import *
from adjoint_solvers import *
from fse_cuda_kernels_runtime import *

# Ensure backward compatibility
__all__ = [
    'FSEField', 'FieldType', 'FieldOperations', 'FSULanguageModel',
    'FSEAdjointSolvers', 'FlowField_FLIT', 'FlowField_FSEBlock'
]