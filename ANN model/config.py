import os
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 42 ## for reproducibility 

TARGET = "Biomass"

OUTPUT_PATH = "outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

VIF_THRESHOLD = 5

N_TRIALS = 4000

VIs = [
    "ARVI", "CCCI", "CSI", "CVI", "Datt99",
    "DVI", "EVI", "GCI", "GNDVI", "MCARIOSAVI",
    "MCARI", "MSAVI", "MSRI", "MTCI", "NDRI",
    "NDVI", "NGRDI", "OSAVI", "RDVI", "RECI",
    "RGI", "RTVI", "SAVI", "SRI",
    "TCARIOSAVI", "TCARI", "TVI", "VARI",
    "EGVI", "LCI"
]

SFs = ["CC", "PH"]

SBs = ["blue", "green", "nir", "red edge", "red"]

TFs = [
    "Contrast",
    "Dissimilarity",
    "Correlation",
    "Energy",
    "Homogeneity",
    "Entropy"
]
