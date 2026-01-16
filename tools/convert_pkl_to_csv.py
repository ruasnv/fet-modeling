# tools/convert_pkl_to_csv.py
import pickle as pkl
import pandas as pd
from pathlib import Path
import sys

# Add project root to path to read settings if needed
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.core.config import settings

def convert_data():
    # Use config paths instead of hardcoding "augmented/augmented_data.pkl"
    # This makes the tool smarter!
    base_dir = settings.get('paths.aug_data_dir')
    if not base_dir:
        # Fallback if config isn't set up for augmentation yet
        base_dir = project_root / "data" / "augmented"
    else:
        base_dir = Path(base_dir)

    pkl_path = base_dir / "augmented_data.pkl"
    csv_path = base_dir / "augmented_data.csv"
    
    if not pkl_path.exists():
        print(f"Error: File not found at {pkl_path}")
        return

    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data_object = pkl.load(f)

    # Handle if the pickle is a DataFrame or a raw object
    if isinstance(data_object, pd.DataFrame):
        df = data_object
    else:
        df = pd.DataFrame(data_object)
    
    print(f"Saving to {csv_path}...")
    df.to_csv(csv_path, index=False)
    print("Done.")

if __name__ == "__main__":
    convert_data()