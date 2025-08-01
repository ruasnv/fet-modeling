# scripts/run_data_processing.py
import os
import sys

# Ensure the project root is in the path for module imports
# This is crucial for `from src.config import settings` to work when running from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing.preprocessor import DataPreprocessor

def run_preprocess_data(force_reprocess=False):
    """
    Orchestrates the data processing pipeline.
    It initializes DataPreprocessor with relevant config data from global settings
    and calls its load_or_process_data method.
    """
    print("--- Starting Data Processing Script ---")

    # Pass the relevant *parts* of the global config data to DataPreprocessor
    # DataPreprocessor will then use these dictionaries internally.
    dp: DataPreprocessor = DataPreprocessor()
    # Call the load_or_process_data method which handles checking existing data,
    # processing, and saving.
    if dp.load_or_process_data(force_reprocess=force_reprocess):
        print("\nData processing pipeline completed successfully.")
        return dp # Return the preprocessor instance for potential further use in the same run if needed
    else:
        print("\nData processing pipeline failed.")
        return None

if __name__ == "__main__":
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