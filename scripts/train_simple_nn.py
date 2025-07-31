# scripts/train_simple_nn.py
import torch
import torch.nn as nn
import os
import yaml  # For loading config
from datetime import datetime

# Import modular components from src
from src.data_processing.preprocessor import DataProcessor
from src.models.simple_nn import SimpleNN
from src.training.simple_nn_trainer import NNTrainer  # Or a more general Trainer
from src.evaluation.evaluator import NNEvaluator
from src.cross_validation.cv_runner import PyTorchCrossValidator  # Renamed
from src.utils.plotter import Plotter
from src.utils.helpers import setup_environment, load_config  # New helper


def main():
    # --- 1. Load Configuration ---
    config = load_config('config/simple_nn_config.yaml')  # Load specific config for SimpleNN
    main_config = load_config('config/main_config.yaml')  # Load main project config (paths)

    # Use config values
    RAW_DATA_PATH = main_config['paths']['raw_data_path']
    PROCESSED_DATA_DIR = main_config['paths']['processed_data_dir']
    TRAINED_MODEL_DIR = main_config['paths']['trained_model_dir']
    REPORT_OUTPUT_DIR = main_config['paths']['report_output_dir']

    CV_FINAL_TEST_SPLIT_RATIO = config['data']['cv_final_test_split_ratio']
    NUM_FOLDS = config['data']['num_folds']

    NN_INPUT_DIM = config['model']['input_dim']
    NN_NUM_EPOCHS = config['training']['num_epochs']
    NN_BATCH_SIZE = config['training']['batch_size']
    NN_LEARNING_RATE = config['training']['learning_rate']
    SKIP_TRAINING_IF_EXISTS = config['flags']['skip_training_if_exists']
    SKIP_PLOTS_IF_EXISTS = config['flags']['skip_plots_if_exists']
    CASES_CONFIG_FOR_PLOTS = config['plots']['cases_config_for_best_worst_plots']  # From config now

    setup_environment(REPORT_OUTPUT_DIR)  # Utility function

    # --- 2. Data Loading and Processing ---
    # This step could be its own script (run_data_processing.py) if data processing is complex and takes long
    # For now, keep it here for this script's completeness
    dp = DataProcessor(RAW_DATA_PATH)
    if not dp.load_processed_data(PROCESSED_DATA_DIR):
        print("Processed data not found, preparing new sets...")
        dp.prepare_cv_and_final_test_sets(
            cv_test_split_ratio=CV_FINAL_TEST_SPLIT_RATIO,
            n_splits_cv=NUM_FOLDS,
            random_state=42,
            save_path=PROCESSED_DATA_DIR
        )
    else:
        print(f"Loaded processed data from {PROCESSED_DATA_DIR}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 3. Run Cross-Validation ---
    # Only run if `config['cv']['run_cv']` is true
    if config['cv']['run_cv']:  # New config flag
        model_config_cv = {
            'name': 'SimpleNN',
            'model_class': SimpleNN,
            'model_params': {'input_dim': NN_INPUT_DIM},
            'trainer_class': NNTrainer,  # Consider making Trainer more generic
            'trainer_params': {'optimizer_params': {'lr': NN_LEARNING_RATE}},
            'evaluator_class': NNEvaluator
        }
        print("\nStarting Cross-Validation...")
        cv_runner = PyTorchCrossValidator(
            device=device,
            criterion=nn.MSELoss(),
            scaler_x=dp.get_scalers()[0],
            scaler_y=dp.get_scalers()[1],
            features_for_model=dp.get_features_for_model()
        )

        summary_df, detailed_df = cv_runner.run_cv(
            model_config=model_config_cv,
            X_cv_scaled=dp.get_cv_data()[0],
            y_cv_scaled=dp.get_cv_data()[1],
            X_cv_original_df=dp.get_cv_data()[2],
            cv_fold_indices=dp.get_cv_data()[3],
            num_epochs=NN_NUM_EPOCHS,
            batch_size=NN_BATCH_SIZE,
            model_save_base_dir=os.path.join(TRAINED_MODEL_DIR, 'k_fold_models'),
            report_output_dir=REPORT_OUTPUT_DIR,
            skip_if_exists=SKIP_TRAINING_IF_EXISTS
        )
        # It's better to make save_cv_results a method in Evaluator or a helper
        # from src.evaluation.evaluator import save_cv_results
        # save_cv_results(summary_df, detailed_df, REPORT_OUTPUT_DIR) # Pass the dataframes explicitly
        print("Cross-Validation complete.")
    else:
        print("Skipping Cross-Validation as per config.")

    # --- 4. Train/Load Final Model for Plotting/General Evaluation ---
    final_model_path = os.path.join(TRAINED_MODEL_DIR, 'final_simple_nn_for_plots.pth')
    model = SimpleNN(input_dim=NN_INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_LEARNING_RATE)
    trainer = NNTrainer(model, device, nn.MSELoss(), optimizer, final_model_path)  # Trainer handles saving/loading

    if SKIP_TRAINING_IF_EXISTS and os.path.exists(final_model_path):
        print(f"  Skipping training for final model: already exists at {final_model_path}")
        model.load_state_dict(torch.load(final_model_path, map_location=device))
        model.eval()
        train_losses, test_losses = [], []  # No new losses if skipped
    else:
        print("\nTraining a final SimpleNN model on the entire CV pool for plotting")
        train_losses, test_losses = trainer.train(
            torch.tensor(dp.get_cv_data()[0], dtype=torch.float32).to(device),
            torch.tensor(dp.get_cv_data()[1], dtype=torch.float32).to(device),
            torch.tensor(dp.get_final_test_data()[0], dtype=torch.float32).to(device),
            torch.tensor(dp.get_final_test_data()[1], dtype=torch.float32).to(device),
            num_epochs=NN_NUM_EPOCHS,
            batch_size=NN_BATCH_SIZE
        )

    # --- 5. Evaluate Final Model ---
    evaluator = NNEvaluator(model, device, *dp.get_scalers())
    print("\nEvaluating final model on test data:")
    eval_results_df = evaluator.evaluate_model(*dp.get_final_test_data()[:2])
    print(eval_results_df)  # Print or save these results

    # --- 6. Generate Plots ---
    plotter_instance = Plotter(dp.get_scalers()[0], dp.get_scalers()[1],
                               dp.get_features_for_model(), device)
    plot_dir = os.path.join(REPORT_OUTPUT_DIR, 'final_simple_nn_model', 'specific_characteristic_plots')
    os.makedirs(plot_dir, exist_ok=True)  # Ensure dir exists before checking

    # You had this check in the function, move it here
    if SKIP_PLOTS_IF_EXISTS and os.path.exists(plot_dir) and len(
            os.listdir(plot_dir)) > 0:  # Check if dir exists and not empty
        print(f"Skipping plot generation: plots already exist in {plot_dir}")
    else:
        print("\nGenerating characteristic plots...")
        plotter_instance.id_vds_characteristics(
            model=model,
            full_original_data_for_plot=dp.get_filtered_original_data(),  # Ensure this is the correct data for plotting
            cases_config_for_best_worst_plots=CASES_CONFIG_FOR_PLOTS,
            model_name="SimpleNN (Final Plotting Model)",
            output_dir=plot_dir
        )
        if train_losses:  # Plot loss curves only if training happened
            trainer.plot_losses(train_losses, test_losses, NN_NUM_EPOCHS,
                                model_name="SimpleNN (Final Plotting Model)",
                                output_dir=plot_dir)
        print("Plot generation complete.")


if __name__ == "__main__":
    main()