# scripts/preprocess_data.py
import os
from src.data_processing.dataProcessor import DataProcessor
from config.data_config import RAW_DATA_PATH, PROCESSED_DATA_DIR, CV_FINAL_TEST_SPLIT_RATIO, NUM_FOLDS

def run_preprocess_data():
    dp = DataProcessor(RAW_DATA_PATH)
    if not dp.load_processed_data(PROCESSED_DATA_DIR):
        print("Processing raw data and preparing CV/final test sets...")
        dp.prepare_cv_and_final_test_sets(
            cv_test_split_ratio=CV_FINAL_TEST_SPLIT_RATIO,
            n_splits_cv=NUM_FOLDS,
            random_state=42,
            save_path=PROCESSED_DATA_DIR
        )
        print(f"Processed data saved to: {PROCESSED_DATA_DIR}")
    else:
        print(f"Processed data loaded from: {PROCESSED_DATA_DIR}")
    return dp

if __name__ == "__main__":
    dp = run_preprocess_data()
    # You might want to print some stats about the loaded data here