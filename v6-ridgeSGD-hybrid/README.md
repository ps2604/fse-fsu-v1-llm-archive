# FlowField FSU LLM: Revolutionary Post-Token Language Processing

FlowField FSU (Functional State Unit) is a cutting-edge language modeling architecture that implements **continuous field-based language modeling**. Departing from traditional discrete state transitions, FSU leverages **Float-Native State Elements (FSE)** and **continuous semantic field evolution** to process language with unprecedented contextual continuity and efficiency.

## 🚀 Key Features

-   **Continuous Semantic Fields**: Implements revolutionary post-token language processing using continuous field evolution.
-   **Adjoint-Based Optimization**: Utilizes ultra-optimized adjoint solvers for precise gradient computation in continuous-time dynamics.
-   **FSE (Float-Native State Elements)**: Core infrastructure designed for high-performance state representation.
-   **Ultra-Optimized CUDA Kernels**: Custom-built runtime kernels for maximum GPU throughput using CuPy.
-   **Auralith Dynamic Regulator (V8)**: Hybrid Morph-SGD optimizer for stable and efficient training of large-scale field models.
-   **Unlimited Context Support**: Architecture designed for extended context scaling without the quadratic bottlenecks of traditional Transformers.
-   **Robust Distributed Training**: Full support for multi-GPU training via NCCL with enhanced field synchronization.

## 📁 Project Structure

-   `adjoint_fsu_model.py`: The core FSU language model architecture.
-   `fsu_training_ultra_optimized.py`: Serialized, ultra-optimized training loop with TensorBoard integration.
-   `adjoint_core_optimized.py`: Optimized FlowField infrastructure and field operations.
-   `fse_cuda_kernels_runtime.py`: Custom CUDA kernels for FSE operations.
-   `adjoint_solvers.py`: High-performance PDE and ODE solvers for field dynamics.
-   `fsu_data_processor.py`: Advanced text processing and field-based data loading.
-   `setup.py`: Standard Python packaging and entry points.

## 🛠️ Installation

### Prerequisites

-   Python 3.9+
-   CUDA Toolkit (compatible with CuPy)
-   NVIDIA GPU with compute capability 7.0+

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ps2604/flowfield-fsu-llm.git
    cd flowfield-fsu-llm
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the package (Developer Mode):**
    ```bash
    pip install -e .
    ```

## 📈 Usage

### Training

To start the ultra-optimized training loop:

```bash
fsu-train --sequence_length 4096 --channels 256 --learning_rate 0.002
```

For multi-GPU training:

```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS fsu_training_ultra_optimized.py --enable_multi_gpu
```

### Verification

To verify the model configuration and field continuity:

```bash
fsu-verify
```

## 🧪 Technical Details

FlowField FSU operates by representing language as a continuous evolution of semantic fields. This approach allows the model to capture long-range dependencies and complex linguistic structures through the lens of physical field dynamics. The use of Adjoint methods enables efficient backpropagation through the continuous-time evolution, bridging the gap between deep learning and classical field theory.

## 🤝 Contributing

This is a personal research project. For inquiries or collaboration, please reach out to:
**Pirassena Sabaratnam** - [auralithco@gmail.com](mailto:auralithco@gmail.com)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
