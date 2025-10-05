# Audio DistilBERT

The implementation of "Audio DistilBERT: A distilled audio BERT for speech representation learning".

> Note:
> I have losses original code and try to rebuild it.

**Author:** [fanfan-yu](https://github.com/fanfan-yu)  
**Date:** 2025.10.05  
**License:** MIT

## 🎯 Overview

Audio DistilBERT is a compact, distilled speech representation learning framework that combines BERT-style transformer architecture with knowledge distillation. It achieves competitive performance to larger models while being more efficient in terms of parameters and inference speed.

### Key Features
- 🔊 **Speech-Specific Design**: Optimized for audio spectrograms with 80-channel mel features
- 🧠 **Knowledge Distillation**: Teacher-student learning from larger pre-trained models
- 🎭 **Masked Acoustic Modeling (MAM)**: Self-supervised learning via spectrogram reconstruction
- ⚡ **Efficient Architecture**: Reduced model size (1.8× smaller) with faster inference (1.6× speedup)
- 🖥️ **Multi-GPU Training**: Distributed training support with automatic gradient accumulation

## 📊 Performance Highlights

| Model | Size Reduction | Speedup | Performance Retention |
|-------|----------------|---------|----------------------|
| Audio DistilBERT vs Teacher | **1.8× smaller** | **1.6× faster** | **>98%** |

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+
pip install torch torchaudio
pip install -e .
```

### Training a Student Model

#### 1. Prepare Your Data
```bash
# Place LibriSpeech features in expected structure
data/
├── libri_fbank80/
│   ├── train-clean-100/
│   ├── dev-clean/
│   └── test-clean/
└── cpc_phone/  # optional for downstream tasks
```

#### 2. Sync dependencies

Install UV package manager
```bash
pip install uv
```

Download the dependencies
```bash
# Initialize environment and install dependencies
uv init
uv sync
```

#### 3. Pre-training Teacher or loading pre-trained model

TODO

#### 4. Start Training
```bash
# Basic distillation training
python main_upstream.py \
    --student_config config/distiller_upstream_fbankBase.yaml \
    --teacher_resume teacher.ckpt \
    --name my_audio_distilbert \
    --multi_gpu

# Resume from checkpoint
python main_upstream.py \
    --student_resume checkpoints/student-states-50000.ckpt \
    --teacher_resume teacher.ckpt \
    --name resumed_training
```

## 📁 Project Structure

```
Audio-DistilBERT/
├── config/                          # Configuration files
│   └── distiller_upstream_fbankBase.yaml
├── dataloader/                      # Data pipeline
│   └── dataloader.py               # Multi-dataset support
├── upstream/                        # Core model implementation
│   ├── model.py                    # Transformer architecture
│   ├── mam.py                      # Masked acoustic modeling
│   ├── runner.py                   # Training orchestration
│   ├── optimization.py             # Custom optimizers
│   └── utils.py                    # Utilities
├── Makefile                        # Build automation
├── main_upstream.py                # Training entry point
└── README.md                       # This file
```

## 🏗️ Architecture Details

### Model Components
- **Input Processing**: 80-channel mel-spectrogram features
- **Transformer Encoder**: Configurable hidden dimensions (768 default) with attention heads
- **Masked Acoustic Modeling**: Reconstructs original spectrograms from masked inputs
- **Knowledge Distillation**: Soft target learning from teacher model outputs

### Training Pipeline
1. **Data Preparation**: LibriSpeech mel-spectrograms with dynamic masking
2. **Teacher Model**: Large pre-trained transformer providing soft targets
3. **Student Training**: Distillation + reconstruction loss optimization
4. **Evaluation**: Downstream task performance validation

## ⚙️ Configuration Options

### Essential Parameters
| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `hidden_size` | Transformer hidden dimension | 768 |
| `num_attention_heads` | Attention head count | 12 |
| `mask_proportion` | Masked frames ratio | 0.15 |
| `learning_rate` | Training learning rate | 2e-4 |
| `batch_size` | Training batch size | 12 |

### Advanced Settings
- **Gradient Accumulation**: Handle large batch sizes across GPUs
- **Warmup Scheduling**: Prevent early training instability
- **Attention Head Pruning**: Customize model size vs. performance trade-offs

## 📈 Expected Performance

### Hardware Requirements
- **GPU**: 1× V100 or equivalent (multi-GPU supported)
- **Memory**: ~8GB for batch_size=12, hidden_size=768
- **Training Time**: ~48 hours for 200k steps on 4× GPUs

### Memory Optimization Tips
- Reduce `batch_size` for limited GPU memory
- Increase `gradient_accumulation_steps` for larger effective batch
- Enable gradient checkpointing for very long sequences

## 🧪 Downstream Usage

### Extracting Representations

TODO

## 🤝 Contributing

We welcome contributions! Please:
1. Check existing [issues](https://github.com/fanfan-yu/Audio-DistilBERT/issues)
2. Create feature branches from `main`
3. Add tests for new functionality
4. Submit pull requests with detailed descriptions

## 📊 Todo List
- [ ] Implement downstream tasks code
- [ ] Create Jupyter Notebook tutorials for beginners
- [ ] Save pre-trained model in remote which can be used directly

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LibriSpeech**: Dataset for pre-training

## ⚙️ Requirements
| Package       | Minimum Version |
|---------------|-----------------|
| Python        | 3.96            |
| PyTorch       | 1.12.0          |
| TorchText     | 0.13.0          |
| NumPy         | 1.26.4          |
| [UV](https://github.com/astral-sh/uv) | Latest          |

