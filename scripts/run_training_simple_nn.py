# scripts/run_training_simple_nn.py
import torch
import torch.nn as nn
import torch.optim as optim  # Import optim for dynamic optimizer creation
import os
import sys
from pathlib import Path
import matplotlib  # Ensure matplotlib is imported for backend settings


# Append the parent directory to the system path to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Use 'Agg' backend for non-interactive plotting
matplotlib.use('Agg')

# Import modular components from src
from src.data_processing.preprocessor import DataPreprocessor
from src.models.simple_nn import SimpleNN
from src.training.simple_nn_trainer import NNTrainer
from src.evaluation.evaluator import NNEvaluator
from src.utils.plotter import Plotter
from src.utils.helpers import setup_environment
from src.config import settings
from src.cross_validation.cv_runner import CrossValidator


def main():
    print("--- Starting Model Training Script ---")
    setup_environment()

    # --- Data Loading and Processing check ---
    dp = DataPreprocessor()
    processed_data_dir = settings.get("paths.processed_data_dir")
    if not dp.load_processed_data(processed_data_dir):
        print(f"Error: Processed data not found in {processed_data_dir}.")
        print("Please run the `run_data_processing.py` script first to generate it.")
        return
    else:
        print(f"Loaded processed data from {processed_data_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    nn_input_dim = settings.get('model_params.input_dim')
    # --- 3. Run Cross-Validation ---
    if settings.get('run_flags.run_cross_validation'):
        criterion_cv_instance = getattr(nn, settings.get("training_params.criterion"))()

        model_config_cv = {
            'name': 'SimpleNN',
            'model_class': SimpleNN,
            'model_params': {'input_dim': nn_input_dim},
            'trainer_class': NNTrainer,
            'trainer_params': {
                'num_epochs': settings.get('training_params.num_epochs'),
                'batch_size': settings.get('training_params.batch_size'),
                'learning_rate': settings.get('training_params.learning_rate'),
                'criterion_name': settings.get('training_params.criterion'),  # Still useful for logging/info
                'optimizer_name': settings.get('training_params.optimizer')  # Still useful for logging/info
            },
            'evaluator_class': NNEvaluator
        }
        print("\nStarting Cross-Validation...")

        cv_runner = CrossValidator(
            device=device,
            criterion=criterion_cv_instance,  # Pass the instantiated criterion object
            scaler_x=dp.get_scalers()[0],
            scaler_y=dp.get_scalers()[1],
            features_for_model=dp.get_features_for_model()
        )

        # These variables are assigned but their values are saved to files by cv_runner.run_cv.
        # The linter might warn about them being unused, but this is expected behavior.
        summary_df, detailed_df = cv_runner.run_cv(
            model_config=model_config_cv,
            X_cv_scaled=dp.get_cv_data()[0],
            y_cv_scaled=dp.get_cv_data()[1],
            X_cv_original_df=dp.get_cv_data()[2],
            cv_fold_indices=dp.get_cv_data()[3],
            model_save_base_dir=Path(settings.get("paths.trained_model_dir")) / 'k_fold_models',
            report_output_dir=Path(settings.get("paths.report_output_dir")),
            skip_if_exists=settings.get("run_flags.skip_training_if_exists"),
        )
        print("Cross-Validation complete.")
        # Save detailed CV metrics to CSV
        cv_metrics_path = Path(settings.get("paths.report_output_dir")) / 'cv_detailed_metrics.csv'
        detailed_df.to_csv(cv_metrics_path, index=False)
        print(f"Detailed CV metrics saved to: {cv_metrics_path}")

    else:
        print("Skipping Cross-Validation as per config.")

    # --- 4. Train/Load Final Model for Plotting/General Evaluation ---
    final_model_filename = settings.get('model_params.final_model_filename')
    final_model_path = Path(settings.get("paths.trained_model_dir")) / final_model_filename

    # Initialize SimpleNN with only input_dim
    model = SimpleNN(input_dim=nn_input_dim).to(device)

    # Instantiate criterion and optimizer for the final model training
    criterion_final_instance = getattr(nn, settings.get("training_params.criterion"))()
    optimizer_final_instance = getattr(optim, settings.get("training_params.optimizer"))(
        model.parameters(), lr=settings.get("training_params.learning_rate")
    )

    # Initialize NNTrainer with the instantiated criterion and optimizer objects
    trainer = NNTrainer(
        model=model,
        device=device,
        criterion=criterion_final_instance,  # Pass the object
        optimizer=optimizer_final_instance,  # Pass the object
        model_save_path=final_model_path
    )

    if settings.get("run_flags.skip_training_if_exists") and final_model_path.exists():
        print(f"  Skipping training for final model: already exists at {final_model_path}")
        # The 'load_model' method is correctly defined in NNTrainer.
        # This is likely a linter warning that can be ignored.
        trainer.load_model()
        train_losses, test_losses = [], []
    else:
        print("\nTraining a final SimpleNN model on the entire CV pool for plotting")
        train_losses, test_losses = trainer.train(
            torch.tensor(dp.get_cv_data()[0], dtype=torch.float32),
            torch.tensor(dp.get_cv_data()[1], dtype=torch.float32),
            torch.tensor(dp.get_final_test_data()[0], dtype=torch.float32),
            torch.tensor(dp.get_final_test_data()[1], dtype=torch.float32),
            num_epochs=settings.get("training_params.num_epochs"),
            batch_size=settings.get("training_params.batch_size"),
        )

    # --- 5. Evaluate Final Model ---
    evaluator = NNEvaluator(model, device, *dp.get_scalers())
    print("\nEvaluating final model on test data:")
    eval_results = evaluator.evaluate_model(
        torch.tensor(dp.get_final_test_data()[0], dtype=torch.float32),
        torch.tensor(dp.get_final_test_data()[1], dtype=torch.float32),
        X_test_original_df=dp.get_final_test_data()[2]
    )
    print("Final Test Set Evaluation Results:")
    for metric, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4e}")
        else:
            print(f"  {metric}: {value}")

    # --- 6. Generate Plots ---
    plots_output_dir = Path(
        settings.get("paths.report_output_dir")) / 'final_model' / 'characteristic_plots'
    os.makedirs(plots_output_dir, exist_ok=True)

    #TODO: Mitigate this logic to run_evaluation_on_model.py
    """
    if settings.get("run_flags.skip_plots_if_exists") and plots_output_dir.exists() and any(plots_output_dir.iterdir()):
        print(f"Skipping plot generation: plots already exist in {plots_output_dir}")
    else:
        print("\nGenerating characteristic plots...")
        plotter_instance.id_vds_characteristics(
            model=model,
            full_original_data_for_plot=dp.get_filtered_original_data(),
            cases_config_for_best_worst_plots=settings.get("plot_cases"),
            model_name="SimpleNN (Final Plotting Model)",
            output_dir=plots_output_dir  # Pass Path object directly
        )
    """
    if train_losses:
        trainer.plot_losses(train_losses, test_losses, settings.get("training_params.num_epochs"),
                            model_name="SimpleNN (Final Plotting Model)",
                            output_dir=plots_output_dir)
    print("Training loss plot complete.")

    print("--- Model Training Finished ---")


if __name__ == "__main__":
    main()
