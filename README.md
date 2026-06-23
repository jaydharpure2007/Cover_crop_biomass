# Cover crop biomass modeling using Machine Learning and Artificial Neural Networks

## Overview

This repository contains Python scripts for cover crop aboveground biomass (AGB) estimation using vegetation indices (VIs), spectral bands (SBs), structural features (SFs), and texture features (TFs) derived from UAV imagery. The workflow includes feature selection using variance inflation factor (VIF), hyperparameter optimization, machine learning (ML) and artificial neural network (ANN) model training, SHAP-based explainability analysis, and export of model outputs.

The repository was developed to support reproducible analysis associated with the manuscript:

> *[Cover Crop Biomass Estimation using UAV-Based Multispectral Feature Fusion and Machine Learning]*

---
## Workflow Overview

<p align="center">
<img src="images/Overall workflow.png" width="1000">
</p>

## ANN Repository Structure

```text
project/
│
├── data/                     # Input datasets
├── outputs/                  # Model outputs and Excel results
├── src/
│   ├── main.py               # Main workflow
│   ├── config.py             # Global configuration variables
│   ├── model.py              # ANN model architecture
│   ├── train.py              # Training, evaluation functions and Optuna optimization function
│   ├── utils.py              # Utility functions
│   ├── explainability.py     # SHAP analysis functions
│
├── requirements.txt
```

---

## ML Repository Structure

```text
project/
│
├── Data/                      # Input datasets (train/test Excel files)
├── outputs/                   # Model outputs (Excel results per experiment)
│
├── src/
│   ├── main.py                # Main workflow pipeline
│   ├── config.py              # Global parameters and settings
│   ├── models.py              # Model factory (RF, SVR, PLSR, XGB)
│   ├── tuning.py              # Hyperparameter tuning (RandomizedSearchCV)
│   ├── evaluation.py          # Cross-validation and evaluation metrics
│   ├── explainability.py      # SHAP-based feature importance
│   ├── utils.py               # Preprocessing + VIF + helper functions
│
├── requirements.txt
```

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
git clone https://github.com/jaydharpure2007/Cover_crop_biomass.git
cd Cover_crop_biomass
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
---

# Workflow Pipeline

1. Load training and testing data  
2. Generate feature combinations  
3. Apply VIF-based feature selection  
4. Standardize features and target variable  
5. Train models using Optuna or RandomizedSearchCV  
6. Perform cross-validation  
7. Train final model on full dataset  
8. Evaluate on training and test sets  
9. Compute SHAP feature importance  
10. Save outputs to Excel files

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
[]
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
