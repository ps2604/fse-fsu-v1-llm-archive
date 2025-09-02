################################################################################
# FSU LLM – FlowField Native – Multi-GPU Training Image
# CUDA 11.8 / cuDNN 8 / NCCL • Ubuntu 22.04 • Python 3.10
# Post-Token Language Processing with Continuous Semantic Field Evolution
################################################################################
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUPY_CACHE_DIR=/tmp/cupy_cache
ENV CUPY_TF32=1
ENV CUPY_ACCELERATORS=cub,cutensor

# NCCL Configuration - Significantly increased timeouts and optimizations
ENV NCCL_DEBUG=INFO
ENV NCCL_SHM_DISABLE=1
ENV NCCL_P2P_DISABLE=0
ENV NCCL_IB_DISABLE=1
ENV NCCL_SOCKET_IFNAME=^docker0,lo
ENV NCCL_BLOCKING_WAIT=1
ENV NCCL_ASYNC_ERROR_HANDLING=1
ENV NCCL_TIMEOUT=7200
ENV NCCL_HEARTBEAT_TIMEOUT_SEC=3600
ENV NCCL_PROGRESS_TIMEOUT_SEC=3600
ENV NCCL_COMM_BLOCKING=1
ENV NCCL_MAX_NCHANNELS=16
ENV NCCL_MIN_NCHANNELS=4
ENV NCCL_CHECKS_DISABLE=0
ENV NCCL_CHECK_POINTERS=1

# CUDA and training optimizations
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV OMP_NUM_THREADS=1
ENV TORCH_NCCL_BLOCKING_WAIT=1
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1
ENV TORCH_DISTRIBUTED_DEBUG=DETAIL

# FSU-specific optimizations
ENV FSU_DATA_CACHE_DIR=/tmp/fsu_data_cache
ENV FSU_CHECKPOINT_ASYNC=1
ENV FSU_LARGE_DATASET_MODE=1

WORKDIR /app

# System packages - handle NCCL installation carefully
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    build-essential \
    git \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openssh-client \
    openssh-server \
    net-tools \
    iputils-ping \
    libnuma1 \
    libnuma-dev && \
    rm -rf /var/lib/apt/lists/*

# Install or upgrade NCCL packages (handle held packages)
RUN apt-get update && \
    (apt-get install -y --no-install-recommends libnccl2 libnccl-dev || \
    apt-get install -y --no-install-recommends --allow-change-held-packages libnccl2 libnccl-dev) && \
    rm -rf /var/lib/apt/lists/*

# Create python symlink
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Copy FSU LLM source code
COPY setup.py .
COPY adjoint_core_optimized.py .
COPY adjoint_components.py .
COPY fse_cuda_kernels_runtime.py .
COPY adjoint_fsu_model.py .
COPY fsu_training_ultra_optimized.py .
COPY adp_data_loader.py .
COPY fse_physics_defaults.py .
COPY adjoint_loss_functions.py .
COPY metrics_fsu.py .
COPY adjoint_compatibility.py .
COPY adjoint_solvers.py .
COPY adp_core.py .

# Install FlowField package for FSU LLM
RUN pip install --no-cache-dir -e .

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/cupy_cache /app/checkpoints /app/outputs /app/fsu_models \
    /tmp/fsu_data_cache /app/shared_data /app/logs && \
    chmod +x /app/fsu_training_ultra_optimized.py && \
    chmod 777 /tmp/fsu_data_cache /app/shared_data

# Validate installation
RUN echo "=== NCCL Status ===" && \
    dpkg -l | grep nccl || echo "No NCCL packages found via dpkg" && \
    python3 -c "import cupy as cp; print(f'CuPy: {cp.__version__}'); print(f'CUDA: {cp.cuda.runtime.runtimeGetVersion()}')" || echo "CuPy test failed" && \
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || echo "PyTorch test failed" && \
    python3 -c "from cupy.cuda import nccl; print(f'NCCL available'); nccl.get_unique_id(); print('NCCL test passed')" || echo "NCCL test failed"

# Set FSU LLM training entrypoint
ENTRYPOINT ["python3", "/app/fsu_training_ultra_optimized.py"]