# scripts/run_training_simple_nn.py
import torch
import torch.nn as nn
import os

# Import modular components from src
from src.data_processing.preprocessor import DataPreprocessor
from src.models.simple_nn import SimpleNN
from src.training.simple_nn_trainer import NNTrainer
from src.evaluation.evaluator import NNEvaluator
from src.cross_validation.cv_runner import PyTorchCrossValidator
from src.utils.plotter import Plotter
from src.utils.helpers import setup_environment
from src.config import settings


def main():
    setup_environment()
    # --- Data Loading and Processing check
    dp = DataPreprocessor()
    if not dp.load_processed_data(settings.get("processed_data_folder")):
        print("Processed data not found.")
    else:
        print(f"Loaded processed data from {settings.get("processed_data_folder")}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 3. Run Cross-Validation ---
    # Only run if `config['cv']['run_cv']` is true
    if settings.get('run_cv'):
        model_config_cv = {
            'name': 'SimpleNN',
            'model_class': SimpleNN,
            'model_params': {'input_dim': settings.get("nn_input_dim")},
            'trainer_class': NNTrainer,  # Consider making Trainer more generic
            'trainer_params': {'optimizer_params': {'lr': settings.get('nn_learning_rate')}},
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
            num_epochs=settings.get("nn_number_epochs"),
            batch_size=settings.get("NN_BATCH_SIZE"),
            model_save_base_dir = os.path.join(settings.get("TRAINED_MODEL_DIR"), 'k_fold_models'),
            report_output_dir=settings.get("REPORT_OUTPUT_DIR"),
            skip_if_exists= settings.get("SKIP_TRAINING_IF_EXISTS")
        )
        # It's better to make save_cv_results a method in Evaluator or a helper
        # from src.evaluation.evaluator import save_cv_results
        # save_cv_results(summary_df, detailed_df, REPORT_OUTPUT_DIR) # Pass the dataframes explicitly
        print("Cross-Validation complete.")
    else:
        print("Skipping Cross-Validation as per config.")

    # --- 4. Train/Load Final Model for Plotting/General Evaluation ---
    final_model_path = os.path.join(settings.get("TRAINED_MODEL_DIR"), 'final_simple_nn_for_plots.pth')
    model = SimpleNN(input_dim=settings.get("NN_INPUT_DIM")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr= settings.get("NN_LEARNING_RATE"))
    trainer = NNTrainer(model, device, nn.MSELoss(), optimizer, final_model_path)  # Trainer handles saving/loading

    if settings.get("SKIP_TRAINING_IF_EXISTS") and os.path.exists(final_model_path):
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
            num_epochs= settings.get("NN_NUM_EPOCHS"),
            batch_size=settings.get("NN_BATCH_SIZE")
        )

    # --- 5. Evaluate Final Model ---
    evaluator = NNEvaluator(model, device, *dp.get_scalers())
    print("\nEvaluating final model on test data:")
    eval_results_df = evaluator.evaluate_model(*dp.get_final_test_data()[:2])
    print(eval_results_df)  # Print or save these results

    # --- 6. Generate Plots ---
    plotter_instance = Plotter(dp.get_scalers()[0], dp.get_scalers()[1],
                               dp.get_features_for_model(), device)
    plot_dir = os.path.join(settings.get("REPORT_OUTPUT_DIR"), 'final_simple_nn_model', 'specific_characteristic_plots')
    os.makedirs(plot_dir, exist_ok=True)  # Ensure dir exists before checking

    # You had this check in the function, move it here
    if settings.get("SKIP_PLOTS_IF_EXISTS") and os.path.exists(plot_dir) and len(
            os.listdir(plot_dir)) > 0:  # Check if dir exists and not empty
        print(f"Skipping plot generation: plots already exist in {plot_dir}")
    else:
        print("\nGenerating characteristic plots...")
        plotter_instance.id_vds_characteristics(
            model=model,
            full_original_data_for_plot=dp.get_filtered_original_data(),  # Ensure this is the correct data for plotting
            cases_config_for_best_worst_plots= settings.get("CASES_CONFIG_FOR_PLOTS"),
            model_name="SimpleNN (Final Plotting Model)",
            output_dir=plot_dir
        )
        if train_losses:  # Plot loss curves only if training happened
            trainer.plot_losses(train_losses, test_losses, settings.get("NN_NUM_EPOCHS"),
                                model_name="SimpleNN (Final Plotting Model)",
                                output_dir=plot_dir)
        print("Plot generation complete.")


if __name__ == "__main__":
    main()