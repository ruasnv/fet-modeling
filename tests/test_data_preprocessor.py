from src.data_processing.preprocessor import DataPreprocessor

# Assuming your config files are in the 'config' directory relative to where you run this.
# Or provide full paths.
dp = DataPreprocessor('config/main_config.yaml', 'config/data_config.yaml')
data_ready = dp.load_or_process_data()

if data_ready:
    print("Data is ready!")
    # You can now access:
    # X_cv, y_cv, X_cv_original_df, cv_fold_indices = dp.get_cv_data()
    # X_test, y_test, X_test_original_df = dp.get_final_test_data()
    # scaler_x, scaler_y = dp.get_scalers()
else:
    print("Failed to prepare data.")