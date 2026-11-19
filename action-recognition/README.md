# 🎬 Action Recognition on UCF101

A video action recognition project based on R3D-18, demonstrating the use of 3D CNNs to model spatiotemporal features.

## 📁 Project Structure

```
action-recognition/
├── src/
│   ├── dataset.py      # UCF101 dataset loading
│   ├── model.py        # R3D-18 model
│   ├── train.py        # Training script
│   └── evaluate.py     # Evaluation script
├── scripts/
│   ├── setup_env.sh    # Environment setup
│   ├── download_data.sh # Dataset download
│   ├── train.sh        # SLURM training job
│   └── evaluate.sh     # SLURM evaluation job
├── configs/
│   └── config.yaml     # Training configuration
├── checkpoints/        # Model weights
└── results/            # Training results
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# On Triton
module load anaconda
bash scripts/setup_env.sh
source activate action_recog
```

### 2. Download the Dataset

```bash
# Option 1: interactive
bash scripts/download_data.sh

# Option 2: submit as a SLURM job
sbatch scripts/download_data.sh
```

### 3. Start Training

```bash
# Submit the training job
sbatch scripts/train.sh

# Check job status
squeue -u $USER

# Follow the log
tail -f results/train_<job_id>.out
```

### 4. Evaluate the Model

```bash
sbatch scripts/evaluate.sh
```

## 📊 Expected Results

| Metric | Expected Value |
|------|--------|
| Top-1 Accuracy | 70-80% |
| Top-5 Accuracy | 90-95% |
| Training Time | 3-4 hours |
| Inference Speed | ~50 FPS |

## 🛠️ Technical Details

### Model Architecture
- **R3D-18**: 18-layer 3D ResNet that uses 3D convolutions to model spatiotemporal features
- **Pretraining**: Kinetics-400 pretrained weights
- **Input**: 16 frames × 112 × 112

### Data Processing
- Uniformly sample 16 frames per clip
- Data augmentation: random flipping, cropping, color jitter
- Normalization: Kinetics-400 mean / std

### Training Strategy
- Optimizer: AdamW
- Learning rate: 0.001 with cosine annealing
- Mixed-precision training (AMP)
- Batch size: 8

## 📝 Relevance to Thesis Topic

This project directly demonstrates video processing and deep learning skills, and is closely related to the master's thesis topic
"Gesture and Facial Expression Recognition":

1. **Video processing**: extracting and processing frame sequences from video
2. **Spatiotemporal modeling**: 3D CNNs capture the spatial and temporal features of actions
3. **Transfer learning**: transferring from a large-scale dataset to a specific task
4. **Extensibility**: the framework can be applied directly to gesture / facial expression recognition

## 📚 References

- [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
- [R3D Paper](https://arxiv.org/abs/1711.11248)
- [Kinetics Dataset](https://www.deepmind.com/open-source/kinetics)
