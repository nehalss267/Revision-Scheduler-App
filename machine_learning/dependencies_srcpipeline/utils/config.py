from pathlib import Path

# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURE_DATA_DIR = DATA_DIR / "features"

# Models
MODEL_DIR = PROJECT_ROOT / "models"

# Random seed for reproducibility
RANDOM_SEED = 42

# Default train/test split
TEST_SIZE = 0.2
