# scripts/run_data_processing.py
import os
import sys

# Append the parent directory to the system path to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing.augmented_processor import AugmentedDataPreprocessor

def run_preprocess_data(force_reprocess=False):
    """
    Orchestrates the data processing pipeline.
    It initializes DataPreprocessor with relevant config data from global settings
    and calls its load_or_process_data method.
    """
    print("Starting Data Processing Script")
    dp: AugmentedDataPreprocessor = AugmentedDataPreprocessor()
    # Call the load_or_process_data method which handles checking existing data, processing, and saving.
    if dp.load_or_process_data(force_reprocess=force_reprocess):
        print("\nData processing pipeline completed successfully.")
        return dp # Return the preprocessor instance
    else:
        print("\nData processing pipeline failed.")
        return None

if __name__ == "__main__":
    # Not necessary for simple run
    # Example of how to use force_reprocess via a simple argument
    import argparse
    parser = argparse.ArgumentParser(description="Run data preprocessing pipeline.")
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing of data even if processed files exist.')
    args = parser.parse_args()

    dp = run_preprocess_data(force_reprocess=args.force_reprocess)
    if dp:
        print("Preprocessing data done.")
    else:
        print("Failed to prepare data.")