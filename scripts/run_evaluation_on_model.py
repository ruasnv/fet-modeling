# scripts/run_evaluation_on_model.py

from pathlib import Path
import pandas as pd
import torch
import os
import sys
import matplotlib
matplotlib.use('Agg')

# Append the parent directory to the system path to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.preprocessor import DataPreprocessor
from src.models.simple_nn import SimpleNN
from src.evaluation.evaluator import NNEvaluator
from src.utils.plotter import Plotter
from src.utils.helpers import setup_environment, determine_characteristic_plot_cases
from src.config import settings


def run_evaluation():
    """
    This script orchestrates the final evaluation of a pre-trained model.
    It loads processed data, loads a saved model, evaluates its performance
    on the final held-out test set, and generates key characteristic plots.
    """
    print("   Starting Model Evaluation Script   ")
    setup_environment()

    processed_data_dir = settings.get('paths.processed_data_dir')
    trained_model_dir = settings.get('paths.trained_model_dir')
    report_output_dir = settings.get('paths.report_output_dir')
    plots_output_dir = Path.joinpath(report_output_dir, 'final_model_evaluation')

    os.makedirs(plots_output_dir, exist_ok=True)

    print("\n Loading Processed Data ")
    dp = DataPreprocessor()
    if not dp.load_processed_data(processed_data_dir):
        print(f"Error: Processed data not found in {processed_data_dir}.")
        print("Please run the `prepare_data.py` script first to generate it.")
        return

    features_for_model = dp.get_features_for_model()
    nn_input_dim = len(features_for_model)

    print("\n Loading Trained Model ")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = Path.joinpath(trained_model_dir, 'final_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}.")
        print("Please run the `train_model.py` script first to train and save the model.")
        return

    print(f"Loading final trained model from: {model_path}")
    model = SimpleNN(input_dim=nn_input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluate on Final Test Set
    print("\n   Evaluating on Final Test Set   ")
    X_test_scaled, y_test_scaled, X_test_original_df = dp.get_final_test_data()
    evaluator = NNEvaluator(model, device, *dp.get_scalers())

    results = evaluator.evaluate_model(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test_scaled, dtype=torch.float32),
        X_test_original_df=X_test_original_df
    )

    print("Final Test Set Evaluation Results:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4e}")
        else:
            print(f"  {metric}: {value}")

    final_metrics_df = pd.DataFrame([results])
    final_metrics_path = Path(report_output_dir) / 'final_model_metrics.csv'
    final_metrics_df.to_csv(final_metrics_path, index=False)
    print(f"Final model metrics saved to: {final_metrics_path}")

    print("\n    Generating Characteristic Plots   ")

    dynamic_plot_cases = determine_characteristic_plot_cases(
        model=model,
        full_filtered_original_df=dp.get_filtered_original_data(),
        scaler_X=dp.get_scalers()[0],
        scaler_y=dp.get_scalers()[1],
        features_for_model=features_for_model,
        device=device
    )

    if not dynamic_plot_cases:
        print("Skipping plot generation: No valid dynamic plot cases could be determined from the data.")
        return

    plotter = Plotter(
        scaler_X=dp.get_scalers()[0],
        scaler_y=dp.get_scalers()[1],
        features_for_model=features_for_model,
        device=device
    )

    plotter.id_vds_characteristics(
        model=model,
        full_original_data_for_plot=dp.get_filtered_original_data(),
        cases_config_for_best_worst_plots= dynamic_plot_cases,
        model_name= "Final SimpleNN Model",
        output_dir=plots_output_dir
    )

    print(f"\nEvaluation and plots complete. Results saved to: {plots_output_dir}")
    print("   Model Evaluation Script Finished   ")


if __name__ == "__main__":
    run_evaluation()