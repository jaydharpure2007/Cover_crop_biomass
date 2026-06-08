"""
Configuration file for ANN biomass modeling workflow.

Contains:
    - Global settings
    - Feature groups
    - Optimization parameters
    - Output configuration
"""
import os
import torch

# --------------------------------------------------
# Computing device
# --------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Global settings
# --------------------------------------------------
SEED = 42
TARGET = "Biomass"

# --------------------------------------------------
# Output configuration
# --------------------------------------------------
OUTPUT_PATH = "outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --------------------------------------------------
# Feature selection
# --------------------------------------------------
VIF_THRESHOLD = 5

# --------------------------------------------------
# Hyperparameter optimization
# --------------------------------------------------
N_TRIALS = 10

# --------------------------------------------------
# Vegetation indices (VIs)
# --------------------------------------------------
VIs = [
    "ARVI", "CCCI", "CSI", "CVI", "Datt99",
    "DVI", "EVI", "GCI", "GNDVI", "MCARIOSAVI",
    "MCARI", "MSAVI", "MSRI", "MTCI", "NDRI",
    "NDVI", "NGRDI", "OSAVI", "RDVI", "RECI",
    "RGI", "RTVI", "SAVI", "SRI",
    "TCARIOSAVI", "TCARI", "TVI", "VARI",
    "EGVI", "LCI"
]

# --------------------------------------------------
# Structural features (SFs)
# --------------------------------------------------
SFs = ["CC", "PH"]

# --------------------------------------------------
# Spectral bands (SBs)
# --------------------------------------------------
SBs = ["blue", "green", "nir", "red edge", "red"]

# --------------------------------------------------
# Texture features (TFs)
# --------------------------------------------------
TFs = [
    "Contrast",
    "Dissimilarity",
    "Correlation",
    "Energy",
    "Homogeneity",
    "Entropy"
]