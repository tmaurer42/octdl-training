# OCTDL Training

This repository contains the code to train and evaluate deep learning models for Optical Coherence Tomography (OCT) image classification using the OCTDL dataset. It supports both centralized training and federated learning approaches (FedAvg and FedBuff).

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running Experiments](#running-experiments)
  - [Centralized Training](#centralized-training)
  - [Federated Learning (FedAvg)](#federated-learning-fedavg)
  - [Federated Learning (FedBuff)](#federated-learning-fedbuff)
- [Evaluating Results](#evaluating-results)
- [Supported Models](#supported-models)
- [Citation](#citation)

## Dataset

This project uses the **OCTDL (OCT Disease Labels)** dataset, a large-scale open-source dataset of labeled Optical Coherence Tomography (OCT) images for retinal disease classification. The dataset is publicly available under the **CC BY 4.0** license.

### About the Dataset

The OCTDL dataset contains OCT images across 7 classes:
- **AMD** - Age-related Macular Degeneration
- **DME** - Diabetic Macular Edema
- **ERM** - Epiretinal Membrane
- **NO** - Normal
- **RAO** - Retinal Artery Occlusion
- **RVO** - Retinal Vein Occlusion
- **VID** - Vitreomacular Interface Disease

### Download

Download the OCTDL dataset from Mendeley Data:  
🔗 **https://data.mendeley.com/datasets/sncdhf53xc/1**

### Dataset Setup

After downloading, place the dataset in the root folder with the following structure:

```
/OCTDL
├── OCTDL_labels.csv
├── AMD/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── DME/
├── ERM/
├── NO/
├── RAO/
├── RVO/
└── VID/
```

### Reference

If you use this dataset, please cite the original paper:

> Kulyabin, M., Zhdanov, A., Nikiforova, A. et al. OCTDL: Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods. *Sci Data* **11**, 365 (2024). https://doi.org/10.1038/s41597-024-03182-7

## Requirements

- **Python 3.9**
- PyTorch 1.13.1
- torchvision 0.14.1
- scikit-learn
- scikit-image
- pandas
- optuna
- flwr[simulation] (Flower framework for federated learning)
- numpy 1.26.4

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tmaurer42/octdl-training.git
   cd octdl-training
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   
   Use the provided installation script which installs PyTorch with the correct version for your OS:
   ```bash
   python install.py
   ```

   This script will:
   - Install PyTorch 1.13.1 with CUDA 11.6 support (Linux) or CPU version (macOS)
   - Install all packages from `requirements.txt`

   **Alternative manual installation:**
   ```bash
   # For macOS
   pip install torch==1.13.1 torchvision==0.14.1
   
   # For Linux with CUDA 11.6
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
   
   # Then install other requirements
   pip install -r requirements.txt
   ```

## Project Structure

```
octdl-training/
├── install.py                      # Installation script
├── requirements.txt                # Python dependencies
│
├── # Centralized Training
├── run_all_centralized_experiments.py  # Run centralized training experiments
├── eval_centralized.py                 # Evaluate centralized models
├── centralized/
│   └── optimization.py                 # Centralized training optimization
│
├── # Federated Learning
├── run_fedavg_trials.py            # Run FedAvg experiments
├── run_fedbuff_trials.py           # Run FedBuff experiments
├── eval_fedavg_trials.py           # Evaluate FedAvg models
├── eval_fedbuff_trials.py          # Evaluate FedBuff models
├── federated_learning/
│   ├── client.py                   # FL client implementation
│   ├── fedavg.py                   # FedAvg strategy
│   ├── fedbuff.py                  # FedBuff strategy
│   ├── optimization.py             # FL optimization
│   ├── simulation.py               # FL simulation
│   ├── strategy.py                 # FL strategy base
│   └── utils.py                    # FL utilities
│
├── # Shared Modules
├── shared/
│   ├── data.py                     # Dataset loading and preprocessing
│   ├── model.py                    # Model architectures
│   ├── training.py                 # Training utilities
│   ├── metrics.py                  # Evaluation metrics
│   └── utils.py                    # General utilities
│
├── # Results
├── results_centralized/            # Centralized training results
├── results_FedAvg/                 # FedAvg results
├── results_FedBuff/                # FedBuff results
├── figures/                        # Generated figures
│
├── evaluate.py                     # General evaluation script
├── Results.ipynb                   # Results analysis notebook
└── README.md
```

## Running Experiments

### Centralized Training

Run centralized training experiments with hyperparameter optimization using Optuna:

```bash
python run_all_centralized_experiments.py
```

This will train the following configurations:
- ResNet18 with transfer learning
- ResNet18 without transfer learning
- MobileNetV2 with transfer learning

**Custom configuration:**

You can modify `run_all_centralized_experiments.py` to change:
- `classes`: Disease classes to classify (e.g., `[OCTDLClass.AMD, OCTDLClass.NO]` for binary AMD detection)
- `model_type`: `'ResNet18'`, `'ResNet50'`, `'MobileNetV2'`, or `'EfficientNetV2'`
- `transfer_learning`: `True` or `False`
- `loss_fn_type`: `'WeightedCrossEntropy'` for imbalanced datasets
- `optimization_mode`: `'maximize_f1_macro'` for F1 score optimization
- `n_jobs`: Number of parallel Optuna trials

### Federated Learning (FedAvg)

Run Federated Averaging experiments:

```bash
python run_fedavg_trials.py
```

**Default configuration:**
- 20 clients
- 260 total updates
- Clients per round: 10, 5, or 3
- Models: ResNet18 (TL), ResNet18 (no TL), MobileNetV2 (TL)

### Federated Learning (FedBuff)

Run Federated Buffered Aggregation experiments:

```bash
python run_fedbuff_trials.py
```

**Default configuration:**
- 20 clients
- 260 total updates
- Buffer sizes: 10, 5, or 3
- Models: ResNet18 (TL), ResNet18 (no TL), MobileNetV2 (TL)

### Multi-class and ResNet50 Experiments

```bash
python run_multiclass_trials.py   # All 7 classes
python run_resnet50_trials.py     # ResNet50 experiments
```

## Evaluating Results

After training, evaluate the models using the corresponding evaluation scripts:

```bash
# Evaluate centralized models
python eval_centralized.py

# Evaluate FedAvg models
python eval_fedavg_trials.py

# Evaluate FedBuff models
python eval_fedbuff_trials.py
```

### Results Analysis

Open the Jupyter notebook for detailed results analysis and visualization:

```bash
jupyter notebook Results.ipynb
```

Results are stored in SQLite databases:
- `results_centralized/results.sqlite3`
- `results_FedAvg/results.sqlite3`
- `results_FedBuff/results.sqlite3`

Model checkpoints are saved in the `checkpoints/` subdirectory of each results folder.

## Supported Models

| Model | Transfer Learning | Description |
|-------|-------------------|-------------|
| ResNet18 | ✓ / ✗ | Lightweight ResNet variant |
| ResNet50 | ✓ / ✗ | Deeper ResNet variant |
| MobileNetV2 | ✓ / ✗ | Efficient mobile architecture |
| EfficientNetV2 | ✓ / ✗ | State-of-the-art efficient model |

All models are adapted for grayscale OCT images (converted to 3-channel for transfer learning compatibility).

## Citation

## Citation

If you use this code, please cite our paper:

> Alam, H. M. T., Maurer, T., Selim, A. M., Eiletz, M., Barz, M., & Sonntag, D. (2026).
> Asynchronous federated learning for web-based OCT image analysis.
> *Journal of Medical Imaging*, 13(1), 014501. https://doi.org/10.1117/1.JMI.13.1.014501

### BibTeX

```bibtex
@article{Alam2026JMIAsyncFLWebOCT,
  title   = {Asynchronous Federated Learning for Web-Based OCT Image Analysis},
  author  = {Alam, Hasan Md Tusfiqur and Maurer, Tim and Selim, Abdulrahman Mohamed and Eiletz, Matthias and Barz, Michael and Sonntag, Daniel},
  journal = {Journal of Medical Imaging},
  year    = {2026},
  volume  = {13},
  number  = {1},
  pages   = {014501},
  doi     = {10.1117/1.JMI.13.1.014501}
}
```




### License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.


### contact
- hasan.alam@dfki.de
