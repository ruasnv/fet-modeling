# main.py
import torch
import torch.nn as nn
import os
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')  # Ensure plots are saved, not displayed interactively
import matplotlib.pyplot as plt
from src.dataProcessor import DataProcessor
from src.pytorchModels import SimpleNN
from src.pytorchTrain import NNTrainer
from src.pytorchEval import NNEvaluator
from src.pytorchCrossVal import PyTorchCrossValidator
from src.plotter import Plotter
from src.helpers import debug_single_prediction
from datetime import datetime


# --- Configuration ---
RAW_DATA_PATH = 'data/nmoshv.csv'
PROCESSED_DATA_DIR = 'data/processed_data'
TRAINED_MODEL_DIR = 'trained_models'
REPORT_OUTPUT_DIR = 'reports/models'
TEMP_FILTER = 27.0

# Data Splitting Configuration
CV_FINAL_TEST_SPLIT_RATIO = 0.1
NUM_FOLDS = 10

# Model Hyperparameters (for SimpleNN)
#TODO CHANGE THE INPUT DIMENSIONS AFTER FEATURE ENGINEERING
NN_INPUT_DIM = 4
NN_NUM_EPOCHS = 100
NN_BATCH_SIZE = 64
NN_LEARNING_RATE = 0.001

# Global setting to skip training if models exist
SKIP_TRAINING_IF_EXISTS = True

# Config for the best/worst case plots
cases_config_for_best_worst_plots = {
    'Cut-off': [
        {'label': 'Best Case (W=0.3µm, L=0.3µm, Vg=0.45V)', 'W': 1e-05, 'L': 3e-07, 'Vg_const': 0.65,
         'Vds_range': [0, 4]},
        {'label': 'Worst Case (W=10µm, L=3µm, Vg=-0.15V)', 'W': 6e-07, 'L': 1e-05, 'Vg_const': -0.05,
         'Vds_range': [0, 4]}
    ],
    'Linear': [
        {'label': 'Best Case (W=0.6µm, L=1.2µm, Vg=2.9V)', 'W': 1e-05, 'L': 3e-07, 'Vg_const': 3.6,
         'Vds_range': [0, 3]},
        {'label': 'Worst Case (W=10µm, L=1µm, Vg=2.0V)', 'W': 3e-07, 'L': 1e-05, 'Vg_const':  1.148,
         'Vds_range': [0, 3]}
    ],
    'Saturation': [
        {'label': 'Best Case (W=10µm, L=0.3µm, Vg=2.25V)', 'W': 1e-05, 'L': 3e-07, 'Vg_const': 3.6,
         'Vds_range': [0, 4]},
        {'label': 'Worst Case (W=10µm, L=10µm, Vg=0.9V)', 'W': 3e-07, 'L': 1e-05, 'Vg_const': 1.148,
         'Vds_range': [0, 4]}
    ]
}

def main():
    # --- Environment Setup ---
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 8
    warnings.filterwarnings('ignore')

    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

    # --- 1. Data Processing and Preparation for CV & Final Test ---
    data_processor = DataProcessor(RAW_DATA_PATH)

    if not data_processor.load_processed_data(PROCESSED_DATA_DIR):
        data_processor.prepare_cv_and_final_test_sets(
            cv_test_split_ratio=CV_FINAL_TEST_SPLIT_RATIO,
            n_splits_cv=NUM_FOLDS,
            random_state=42,
            save_path=PROCESSED_DATA_DIR
        )

    X_cv_scaled, y_cv_scaled, X_cv_original_df, cv_fold_indices = data_processor.get_cv_data()
    X_final_test_scaled, y_final_test_scaled, X_final_test_original_df = data_processor.get_final_test_data()
    scaler_X, scaler_y = data_processor.get_scalers()
    features_for_model = data_processor.get_features_for_model()
    full_filtered_original_df = data_processor.get_filtered_original_data()

    #2. Define SimpleNN for Cross-Validation
    pytorch_model_config = {
        'name': 'SimpleNN',
        'model_class': SimpleNN,
        'model_params': {'input_dim': NN_INPUT_DIM},
        'trainer_class': NNTrainer,
        'trainer_params': {'optimizer_params': {'lr': NN_LEARNING_RATE}},
        'evaluator_class': NNEvaluator,
    }

    #3. Run K-Fold Cross-Validation for SimpleNN
    all_summary_cv_results = pd.DataFrame()
    all_detailed_cv_results = pd.DataFrame()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f"Using device: {device}")

    pytorch_cv_runner = PyTorchCrossValidator(
        device=device,
        criterion=nn.MSELoss(),
        scaler_x=scaler_X,
        scaler_y=scaler_y,
        features_for_model=features_for_model
    )

    print(f"\n")
    print(f"Running CV for PyTorch Model: {pytorch_model_config['name']}")
    print(f"***")

    summary_df, detailed_df = pytorch_cv_runner.run_cv(
        model_config=pytorch_model_config,
        X_cv_scaled=X_cv_scaled,
        y_cv_scaled=y_cv_scaled,
        X_cv_original_df=X_cv_original_df,
        cv_fold_indices=cv_fold_indices,
        num_epochs=NN_NUM_EPOCHS,
        batch_size=NN_BATCH_SIZE,
        model_save_base_dir=os.path.join(TRAINED_MODEL_DIR, 'k_fold_models'),
        report_output_dir=REPORT_OUTPUT_DIR,
        skip_if_exists=SKIP_TRAINING_IF_EXISTS
    )
    all_summary_cv_results = pd.concat([all_summary_cv_results, summary_df], ignore_index=True)
    all_detailed_cv_results = pd.concat([all_detailed_cv_results, detailed_df], ignore_index=True)

    # --- Save Overall Cross-Validation Results ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_csv_path = os.path.join(REPORT_OUTPUT_DIR, f'cv_summary_results_{timestamp}.csv')
    detailed_csv_path = os.path.join(REPORT_OUTPUT_DIR, f'cv_detailed_results_{timestamp}.csv')

    summary_csv_formatter = {
        'R2_log_Avg': '{:.4f}'.format, 'R2_log_Std': '{:.4f}'.format,
        'MAE_Orig_Avg': '{:.4e}'.format, 'MAE_Orig_Std': '{:.4e}'.format,
        'RMSE_Orig_Avg': '{:.4e}'.format, 'RMSE_Orig_Std': '{:.4e}'.format,
        'MAPE_Orig_Avg': '{:.4f}'.format, 'MAPE_Orig_Std': '{:.4f}'.format
    }
    detailed_csv_formatter = {
        'R2_log': '{:.4f}'.format,
        'MAE_original': '{:.4e}'.format,
        'RMSE_original': '{:.4e}'.format,
        'MAPE_original': '{:.4f}'.format
    }

    formatted_summary_str = all_summary_cv_results.to_string(index=False, formatters=summary_csv_formatter)
    with open(summary_csv_path, 'w') as f:
        f.write(formatted_summary_str)

    formatted_detailed_str = all_detailed_cv_results.to_string(index=False, formatters=detailed_csv_formatter)
    with open(detailed_csv_path, 'w') as f:
        f.write(formatted_detailed_str)

    print("\n")
    print("Overall Cross-Validation Results Summary")
    print("***")

    console_formatter = {
        'R2_log_Avg': '{:.4f}'.format, 'R2_log_Std': '{:.4f}'.format,
        'MAE_Orig_Avg': '{:.4e}'.format, 'MAE_Orig_Std': '{:.4e}'.format,
        'RMSE_Orig_Avg': '{:.4e}'.format, 'RMSE_Orig_Std': '{:.4e}'.format,
        'MAPE_Orig_Avg': '{:.4f}'.format, 'MAPE_Orig_Std': '{:.4f}'.format
    }
    print(all_summary_cv_results.to_string(index=False, formatters=console_formatter))

    print(f"\nOverall CV summary saved to: {summary_csv_path}")
    print(f"Overall CV detailed results saved to: {detailed_csv_path}")
    print("***")

    # --- 4. Train Final SimpleNN Model on CV Pool for Plotting ---
    final_model_path = os.path.join(TRAINED_MODEL_DIR, 'final_simple_nn_for_plots.pth')
    final_model = SimpleNN(input_dim=NN_INPUT_DIM).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=NN_LEARNING_RATE)
    final_trainer = NNTrainer(final_model, device, nn.MSELoss(), final_optimizer, final_model_path)

    if SKIP_TRAINING_IF_EXISTS and os.path.exists(final_model_path):
        print(f"  Skipping training for final plotting model: Model already exists at {final_model_path}")
        final_model.load_state_dict(torch.load(final_model_path, map_location=device))
        final_model.eval()
        final_train_losses = []
        final_test_losses = []
    else:
        print("\nTraining a final SimpleNN model on the entire CV pool for plotting")
        X_cv_tensor = torch.tensor(X_cv_scaled, dtype=torch.float32).to(device)
        y_cv_tensor = torch.tensor(y_cv_scaled, dtype=torch.float32).to(device)
        X_final_test_tensor = torch.tensor(X_final_test_scaled, dtype=torch.float32).to(device)
        y_final_test_tensor = torch.tensor(y_final_test_scaled, dtype=torch.float32).to(device)

        final_train_losses, final_test_losses = final_trainer.train(
            X_cv_tensor, y_cv_tensor,
            X_final_test_tensor, y_final_test_tensor,
            num_epochs=NN_NUM_EPOCHS, batch_size=NN_BATCH_SIZE
        )

    # --- DEBUGGING STEP: Test a single prediction ---
    debug_single_prediction(final_model, scaler_X, scaler_y, features_for_model, device)

    # --- 5. Measure Performance of Final SimpleNN Model on Final Test Data ---
    final_evaluator = NNEvaluator(final_model, device, scaler_X, scaler_y, features_for_model)
    print("\n--- Evaluating Final SimpleNN Model on the Held-Out Final Test Set ---")
    final_r2, final_mae_orig, final_rmse_orig, final_mape_orig = \
        final_evaluator.evaluate_model(X_final_test_scaled, y_final_test_scaled)

    # Plot losses for the final model (only if training occurred)
    final_model_report_dir = os.path.join(REPORT_OUTPUT_DIR, 'final_simple_nn_model')
    os.makedirs(final_model_report_dir, exist_ok=True)
    if final_train_losses:
        final_trainer.plot_losses(
            final_train_losses, final_test_losses, NN_NUM_EPOCHS,
            model_name="SimpleNN (Final Plotting Model)",
            output_dir=final_model_report_dir
        )

    # --- 6. Plot Specific FET Characteristics from Final Test Set ---
    print("\n--- Generating SPECIFIC FET characteristic plots with the final SimpleNN model ---")
    plotter = Plotter(scaler_X, scaler_y, features_for_model,
                      device)
    plotter.id_vds_characteristics(
        model=final_model,
        full_original_data_for_plot=full_filtered_original_df,
        specific_cases_config=cases_config_for_best_worst_plots,
        model_name="SimpleNN (Final Plotting Model)",
        output_dir=os.path.join(final_model_report_dir, 'specific_characteristic_plots')
    )


if __name__ == "__main__":
    main()
