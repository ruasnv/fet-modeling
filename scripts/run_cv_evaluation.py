# scripts/run_cv_evaluation.py (a new script)
from pathlib import Path
import torch
import os
import sys
import matplotlib
matplotlib.use('Agg')

# Append the parent directory to the system path to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.simple_nn_trainer import NNTrainer
from src.data_processing.preprocessor import DataPreprocessor
from src.models.simple_nn import SimpleNN
from src.evaluation.evaluator import NNEvaluator
from src.utils.helpers import setup_environment
from src.config import settings
from src.cross_validation.cv_runner import CrossValidator

def run_cv_evaluation():
    print("   Starting Cross-Validation Evaluation Script   ")
    setup_environment()

    processed_data_dir = settings.get('paths.processed_data_dir')
    report_output_dir = settings.get('paths.report_output_dir')
    os.makedirs(report_output_dir, exist_ok=True)

    # Load data and cv splits
    dp = DataPreprocessor()
    if not dp.load_processed_data(processed_data_dir):
        print("Error: Processed data not found. Please run prepare_data.py.")
        return

    X_cv_scaled, y_cv_scaled, X_cv_original_df, cv_fold_indices = dp.get_cv_data()
    nn_input_dim = settings.get('model_params.input_dim')
    # Define your model configuration
    model_config = {
        'name': 'SimpleNN',
        'model_class': SimpleNN,
        'model_params': {'input_dim': nn_input_dim},
        'trainer_class': NNTrainer,
        'trainer_params': {
            'num_epochs': settings.get('training_params.num_epochs'),
            'batch_size': settings.get('training_params.batch_size'),
            'learning_rate': settings.get('training_params.learning_rate'),
            'criterion_name': settings.get('training_params.criterion'),  # Only useful for logging
            'optimizer_name': settings.get('training_params.optimizer')
        },
        'evaluator_class': NNEvaluator
    }

    trained_model_dir= settings.get("paths.trained_model_dir")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cv_runner = CrossValidator(device, None, *dp.get_scalers(), dp.get_features_for_model())

    # Call the new evaluation function
    summary_df, detailed_df = cv_runner.evaluate_fold_models(
        model_config=model_config,
        X_cv_scaled=X_cv_scaled,
        y_cv_scaled=y_cv_scaled,
        X_cv_original_df=X_cv_original_df,
        cv_fold_indices=cv_fold_indices,
        model_save_base_dir=trained_model_dir
    )

    # Save the results to CSV
    summary_df.to_csv(Path(report_output_dir) / f"{model_config['name'].replace(' ', '_').lower()}_cv_summary.csv",
                      index=False)
    detailed_df.to_csv(Path(report_output_dir) / f"{model_config['name'].replace(' ', '_').lower()}_cv_detailed.csv",
                       index=False)

    print("\nCross-validation evaluation complete.")


if __name__ == "__main__":
    run_cv_evaluation()