# main.py
import torch
import torch.nn as nn
import os
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
from config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, TRAINED_MODEL_DIR, REPORT_OUTPUT_DIR,
    CV_FINAL_TEST_SPLIT_RATIO, NUM_FOLDS,
    NN_INPUT_DIM, NN_NUM_EPOCHS, NN_BATCH_SIZE, NN_LEARNING_RATE,
    SKIP_TRAINING_IF_EXISTS, SKIP_PLOTS_IF_EXISTS,
    cases_config_for_best_worst_plots
)

#TODO: POLISH TERMINAL OUTPUT

#Functions
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
    print("\nOverall Cross-Validation Results Summary")
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


#TODO: ADD IF PATH DOESN'T EXIST LOGIC, SO IT İS NOT GENERATED EVERY TIME!!!!
def save_cv_results(summary_df, detailed_df, report_path=REPORT_OUTPUT_DIR):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_csv = os.path.join(report_path, f'cv_summary_results_{timestamp}.csv')
    detailed_csv = os.path.join(report_path, f'cv_detailed_results_{timestamp}.csv')

    # Define custom float formatters for CSV export
    # These formatters match the keys used in the DataFrames (e.g., 'R2_log_Avg', 'MAE_Orig_Avg')
    summary_csv_formatter = {
        'R2_log_Avg': '{:.4f}'.format, 'R2_log_Std': '{:.4f}'.format,
        'MAE_Orig_Avg': '{:.4e}'.format, 'MAE_Orig_Std': '{:.4e}'.format,
        'RMSE_Orig_Avg': '{:.4e}'.format, 'RMSE_Orig_Std': '{:.4e}'.format,
        'MAPE_Orig_Avg': '{:.4f}'.format, 'MAPE_Orig_Std': '{:.4f}'.format
    }
    # These formatters match the keys used in the detailed_df (e.g., 'r2_log', 'mae_orig')
    detailed_csv_formatter = {
        'r2_log': '{:.4f}'.format,  # Changed to 'r2_log'
        'mae_log': '{:.4f}'.format,
        'rmse_log': '{:.4f}'.format,
        'mae_orig': '{:.4e}'.format,  # Changed to 'mae_orig'
        'rmse_orig': '{:.4e}'.format,  # Changed to 'rmse_orig'
        'mape_orig': '{:.4f}'.format  # Changed to 'mape_orig'
    }

    # Format DataFrames to string and save to CSV
    formatted_summary_str = summary_df.to_string(index=False, formatters=summary_csv_formatter)
    with open(summary_csv, 'w') as f:
        f.write(formatted_summary_str)

    formatted_detailed_str = detailed_df.to_string(index=False, formatters=detailed_csv_formatter)
    with open(detailed_csv, 'w') as f:
        f.write(formatted_detailed_str)

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

    print("\nTraining a final SimpleNN model on the entire CV pool for plotting")
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

    # Call id_vds_characteristics func for plots
    plotter.id_vds_characteristics(
        model=model,
        full_original_data_for_plot=dp.get_filtered_original_data(),
        cases_config_for_best_worst_plots=cases_config_for_best_worst_plots,
        model_name="SimpleNN (Final Plotting Model)",
        output_dir=plot_path
    )


def main():
    setup_environment()
    dp = load_or_process_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #TODO ONLY RUN THIS WHEN THE FILES DON'T EXIST, CHECK İNSİDE FUNCTİON
    # Run CV and save results
    summary_df, detailed_df = run_cross_validation(dp, device)
    #save_cv_results(summary_df, detailed_df)

    # Train final model
    final_model, train_losses, test_losses = train_or_load_final_model(
        X_train=dp.get_cv_data()[0],
        y_train=dp.get_cv_data()[1],
        X_test=dp.get_final_test_data()[0],
        y_test=dp.get_final_test_data()[1],
        device=device
    )

    # Evaluate
    # NNEvaluator init signature changed: no features_for_model
    evaluator = NNEvaluator(final_model, device, *dp.get_scalers())
    evaluator.evaluate_model(*dp.get_final_test_data()[:2])

    # Optional debug
    #debug_single_prediction(final_model, *dp.get_scalers(), dp.get_features_for_model(), device)

    plot_if_needed(NNTrainer(final_model, device, nn.MSELoss(), None, ""), train_losses, test_losses)
    generate_characteristic_plots(final_model, dp, device)


if __name__ == "__main__":
    main()
