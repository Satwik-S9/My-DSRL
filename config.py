import torch

# Data Paths
DATA_DIR: str = "data/lyft"
DATA_A_DIR: str = "data/lyft/dataA/dataA"
DATA_B_DIR: str = "data/lyft/dataB/dataB"
DATA_C_DIR: str = "data/lyft/dataC/dataC"
DATA_D_DIR: str = "data/lyft/dataD/dataD"
DATA_E_DIR: str = "data/lyft/dataE/dataE"
MODEL_DIR: str = "models/"

# HyperParameters
BATCH_SIZE: int = 16
NUM_CLASSES: int = 13
NUM_WORKERS: int = 6
PIN_MEMORY: int = True
LR: int = 0.001
BETAS: tuple = (0.9, 0.98)

# General Configs
NUM_EPOCHS: int = 10
IMG_SIZE: int = 128
VAL_SIZE: float = 0.1
TEST_SIZE: float = 0.1
TRAIN_SIZE: float = 1- VAL_SIZE - TEST_SIZE
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 42