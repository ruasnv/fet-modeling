# tests/check_data_loading.py
import sys
import os
from pathlib import Path

# Append the project root to sys.path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.processor import DataPreprocessor

def test_pipeline():
    print("--- Testing Data Pipeline ---")
    
    # Initialize processor (automatically reads config)
    dp = DataPreprocessor() 
    
    # force_reprocess=True ensures we test the math, not just loading a pickle
    data_ready = dp.load_or_process_data(force_reprocess=True)

    if data_ready:
        print("\n SUCCESS: Data Pipeline is working!")
        
        # Verify data shapes
        X_cv, y_cv, _, _ = dp.get_cv_data()
        X_test, y_test, _ = dp.get_final_test_data()
        
        print(f"   CV Data Shape: {X_cv.shape}")
        print(f"   Test Data Shape: {X_test.shape}")
    else:
        print("\n FAILED: Could not prepare data.")

if __name__ == "__main__":
    test_pipeline()