#!/usr/bin/env python3
"""
FlowField FSU LLM Training Setup
===============================

Setup script for FlowField post-token language processing with FSU architecture.
Implements revolutionary continuous field-based language modeling.
"""

from setuptools import setup, find_packages
import os

# Read version from environment or default
version = os.environ.get('FSU_VERSION', '1.0.0')

long_description = "FlowField FSU: Revolutionary post-token language processing using Float-Native State Elements (FSE) and continuous semantic field evolution"
if os.path.exists('README.md'):
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="flowfield-fsu-llm",
    version=version,
    author="Pirassena Sabaratnam",
    author_email="auralithco@gmail.com",
    description="FlowField FSU: Post-Token Language Processing with Continuous Semantic Fields",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="your_repository_url_here",
    
    # FSU LLM Modules
    py_modules=[
        # Core FlowField Infrastructure (reused from FLUXA)
        'adjoint_core_optimized',
        'adjoint_components', 
        'fse_cuda_kernels_runtime',
        
        # FSU Language Model Components
        'adjoint_fsu_model',
        'fsu_data_processor',
        'fsu_async_data_loader',
        'adjoint_loss_functions',
        'fsu_training_ultra_optimized',
        'metrics_fsu',
        'adjoint_solvers',
        'adjoint_compatibility',
        
        # Verification and utilities
        'pre_training_verification'
    ],
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Physics",  # For continuous field processing
    ],
    
    python_requires=">=3.9",
    
    install_requires=[
        "cupy-cuda11x>=12.0.0",
        "numpy>=1.24.0,<1.27.0",
        "google-cloud-storage>=2.10.0,<3.0.0",
        "regex>=2023.5.5",
        "unidecode>=1.3.6",
        "charset-normalizer>=3.1.0",
        "jsonlines>=3.1.0",
    ],
    
    extras_require={
        "dev": [
            "pytest",
            "flake8", 
            "mypy",
            "black",  # Code formatting
        ],
        "validation": [
            "nltk",  # For additional text validation
            "sacrebleu",  # For BLEU score computation
        ],
    },
    
    entry_points={
        "console_scripts": [
            "fsu-train=fsu_training_ultra_optimized:fsu_ultra_optimized_training_loop",
            "fsu-verify=pre_training_verification:main",
        ],
    },
    
    keywords=[
        "machine-learning",
        "language-model",
        "continuous-fields", 
        "post-token-processing",
        "gpu-computing",
        "fse",
        "fsu",
        "flowfield",
        "cupy",
        "nlp",
        "neural-language-processing",
        "semantic-fields",
        "unlimited-context"
    ],
    
    project_urls={
        "Bug Reports": "your_repo_url_here/issues",
        "Source": "your_repo_url_here",
        "Documentation": "your_repo_url_here/docs",
    },
    
    include_package_data=True,
    zip_safe=False,
)