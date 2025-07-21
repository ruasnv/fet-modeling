# main.py
import torch
import torch.nn as nn
import os
import warnings
import matplotlib
matplotlib.use('Agg')  # Ensure plots are saved, not displayed interactively
import matplotlib.pyplot as plt
import pandas as pd
from src.dataProcessor import DataProcessor
from src.pytorchModels import SimpleNN
from src.pytorchTrain import NNTrainer
from src.pytorchEval import NNEvaluator
from src.pytorchCrossVal import PyTorchCrossValidator
from src.plotter import Plotter
from src.helpers import debug_single_prediction
from datetime import datetime
from config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, TRAINED_MODEL_DIR, REPORT_OUTPUT_DIR,
    CV_FINAL_TEST_SPLIT_RATIO, NUM_FOLDS,
    NN_INPUT_DIM, NN_NUM_EPOCHS, NN_BATCH_SIZE, NN_LEARNING_RATE,
    SKIP_TRAINING_IF_EXISTS, SKIP_PLOTS_IF_EXISTS,
    cases_config_for_best_worst_plots
)

def main():
    setup_environment()
    dp = load_or_process_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run CV and save results
    summary_df, detailed_df = run_cross_validation(dp, device)
    save_cv_results(summary_df, detailed_df)

    # Train final model
    final_model, train_losses, test_losses = train_or_load_final_model(
        *dp.get_cv_data()[:2], *dp.get_final_test_data()[:2], device
    )

    # Evaluate
    evaluator = NNEvaluator(final_model, device, *dp.get_scalers(), dp.get_features_for_model())
    print("\n--- Evaluating Final SimpleNN Model on Held-Out Final Test Set ---")
    evaluator.evaluate_model(*dp.get_final_test_data()[:2])

    # Optional debug and plotting
    debug_single_prediction(final_model, *dp.get_scalers(), dp.get_features_for_model(), device)
    plot_if_needed(NNTrainer(final_model, device, nn.MSELoss(), None, ""), train_losses, test_losses)
    generate_characteristic_plots(final_model, dp, device)


if __name__ == "__main__":
    main()

#Functions
#*********************************************
def setup_environment():
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 8
    warnings.filterwarnings('ignore')
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)


def load_or_process_data():
    dp = DataProcessor(RAW_DATA_PATH)
    if not dp.load_processed_data(PROCESSED_DATA_DIR):
        dp.prepare_cv_and_final_test_sets(
            cv_test_split_ratio=CV_FINAL_TEST_SPLIT_RATIO,
            n_splits_cv=NUM_FOLDS,
            random_state=42,
            save_path=PROCESSED_DATA_DIR
        )
    return dp


def run_cross_validation(dp, device):
    model_config = {
        'name': 'SimpleNN',
        'model_class': SimpleNN,
        'model_params': {'input_dim': NN_INPUT_DIM},
        'trainer_class': NNTrainer,
        'trainer_params': {'optimizer_params': {'lr': NN_LEARNING_RATE}},
        'evaluator_class': NNEvaluator
    }

    cv_runner = PyTorchCrossValidator(
        device=device,
        criterion=nn.MSELoss(),
        scaler_x=dp.get_scalers()[0],
        scaler_y=dp.get_scalers()[1],
        features_for_model=dp.get_features_for_model()
    )

    summary_df, detailed_df = cv_runner.run_cv(
        model_config=model_config,
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
    return summary_df, detailed_df


def save_cv_results(summary_df, detailed_df, report_path=REPORT_OUTPUT_DIR):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_csv = os.path.join(report_path, f'cv_summary_results_{timestamp}.csv')
    detailed_csv = os.path.join(report_path, f'cv_detailed_results_{timestamp}.csv')

    summary_df.to_csv(summary_csv, index=False)
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"\nSaved CV summary to: {summary_csv}")
    print(f"Saved detailed CV to: {detailed_csv}")


def train_or_load_final_model(X_train, y_train, X_test, y_test, device):
    model_path = os.path.join(TRAINED_MODEL_DIR, 'final_simple_nn_for_plots.pth')
    model = SimpleNN(input_dim=NN_INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_LEARNING_RATE)
    trainer = NNTrainer(model, device, nn.MSELoss(), optimizer, model_path)

    if SKIP_TRAINING_IF_EXISTS and os.path.exists(model_path):
        print(f"  Skipping training for final model: already exists at {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, [], []

    train_losses, test_losses = trainer.train(
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.float32).to(device),
        torch.tensor(X_test, dtype=torch.float32).to(device),
        torch.tensor(y_test, dtype=torch.float32).to(device),
        num_epochs=NN_NUM_EPOCHS,
        batch_size=NN_BATCH_SIZE
    )
    return model, train_losses, test_losses


def plot_if_needed(trainer, train_losses, test_losses):
    plot_dir = os.path.join(REPORT_OUTPUT_DIR, 'final_simple_nn_model')
    os.makedirs(plot_dir, exist_ok=True)
    if train_losses:  # Skip if empty (i.e., model loaded not trained)
        trainer.plot_losses(
            train_losses, test_losses, NN_NUM_EPOCHS,
            model_name="SimpleNN (Final Plotting Model)",
            output_dir=plot_dir
        )


def generate_characteristic_plots(model, dp, device):
    plotter = Plotter(dp.get_scalers()[0], dp.get_scalers()[1],
                      dp.get_features_for_model(), device)
    plot_path = os.path.join(REPORT_OUTPUT_DIR, 'final_simple_nn_model', 'specific_characteristic_plots')

    if SKIP_PLOTS_IF_EXISTS and os.path.exists(plot_path):
        print(f"Skipping plot generation: plots already exist in {plot_path}")
        return

    plotter.id_vds_characteristics(
        model=model,
        full_original_data_for_plot=dp.get_filtered_original_data(),
        specific_cases_config=cases_config_for_best_worst_plots,
        model_name="SimpleNN (Final Plotting Model)",
        output_dir=plot_path
    )