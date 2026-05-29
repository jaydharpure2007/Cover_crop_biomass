# Cover Crop Biomass Modeling Using Machine Learning and Artificial Neural Networks

## Overview

This repository contains Python scripts for cover crop aboveground biomass (AGB) estimation using vegetation indices (VIs), spectral bands (SBs), structural features (SFs), and texture features (TFs) derived from UAV imagery. The workflow includes feature selection using variance inflation factor (VIF), hyperparameter optimization, machine learning (ML) and artificial neural network (ANN) model training, SHAP-based explainability analysis, and export of model outputs.

The repository was developed to support reproducible analysis associated with the manuscript:

> *[Cover Crop Biomass Estimation using UAV-Based Multispectral Feature Fusion and Machine Learning]*

---

## Repository Structure

```text
project/
│
├── data/                     # Input datasets
├── outputs/                  # Model outputs and Excel results
├── src/
│   ├── main.py               # Main workflow
│   ├── config.py             # Global configuration variables
│   ├── model.py              # ANN model architecture
│   ├── train.py              # Training and evaluation functions
│   ├── utils.py              # Utility functions
│   ├── explainability.py     # SHAP analysis functions
│   └── objective.py          # Optuna optimization function
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Features

* Variance Inflation Factor (VIF)-based feature selection
* Feature fusion analysis using:

  * Vegetation indices (VIs)
  * Structural features (SFs)
  * Spectral bands (SBs)
  * Texture features (TFs)
* Artificial Neural Network (ANN) modeling using PyTorch
* Hyperparameter optimization using Optuna
* SHAP-based feature importance analysis
* Cross-validation and independent test evaluation
* Export of results to Excel format

---

## Software and Hardware Environment

### Software

* Python 3.13.2
* NumPy 2.2.5
* Pandas 2.2.3
* Rasterio 1.5.0
* Statsmodels 0.14.4
* Scikit-learn 1.6.1
* XGBoost 3.0.2
* SHAP 0.50.0
* PyTorch 2.7.1
* Optuna 4.x

### Hardware

* NVIDIA RTX 5000 Ada Generation GPU
* CUDA 11.8
* 32 GB GPU VRAM
* 256 GB system RAM

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Workflow

Run the main script:

```bash
python src/main.py
```

The workflow performs:

1. Data preprocessing
2. VIF-based feature selection
3. Data standardization
4. Hyperparameter optimization using Optuna
5. ANN model training and evaluation
6. SHAP explainability analysis
7. Export of results

---

## Reproducibility

To improve reproducibility:

* Fixed random seeds were applied for NumPy and PyTorch operations
* Deterministic CUDA settings were enabled
* Hyperparameter optimization used a fixed Optuna sampler seed

Example seed setting:

```python
SEED = 42
```

---

## Input Data

The workflow expects Excel datasets containing:

* Predictor variables
* Biomass target variable

Example:

```text
AGB_train_test.xlsx
├── train
└── test
```

---

## Outputs

The workflow generates:

* Model performance metrics
* Predicted vs observed values
* Best hyperparameters
* SHAP feature importance
* Trial optimization history
* Processing time summaries

Outputs are saved as Excel files inside the `outputs/` directory.

---

## Citation

If you use this repository, please cite:

```text
[Add manuscript citation here after publication]
```

---

## License

This project is licensed under the MIT License.
