# src/cross_validation/cv_runner.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from pathlib import Path



class CrossValidator:
    def __init__(self, device, criterion, scaler_x, scaler_y, features_for_model):
        self.device = device
        self.criterion = criterion
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.features_for_model = features_for_model

    def run_cv(self, model_config, X_cv_scaled, y_cv_scaled, X_cv_original_df, cv_fold_indices,
               num_epochs=100, batch_size=64, model_save_base_dir="trained_models/k_fold_models",
               report_output_dir="reports/models",
               skip_if_exists=True):
        """
        Runs K-fold cross-validation for a PyTorch model.

        Args:
            model_config (dict): Configuration for the model, trainer, and evaluator.
            X_cv_scaled (np.ndarray): Full CV pool features, scaled.
            y_cv_scaled (np.ndarray): Full CV pool target, scaled (log_Id).
            X_cv_original_df (pd.DataFrame): Original DataFrame for stratification verification.
            cv_fold_indices (list): List of (train_idx, test_idx) tuples for each fold.
            num_epochs (int): Number of training epochs per fold.
            batch_size (int): Batch size for training.
            model_save_base_dir (Path): Base directory to save models for each fold.
            report_output_dir (Path): Directory to save plots and detailed metrics.
            skip_if_exists (bool): If True, skips training if a saved model for the fold exists.

        Returns:
            pd.DataFrame: A DataFrame containing average and std deviation of metrics across folds.
            pd.DataFrame: A DataFrame with detailed metrics for each fold.
        """
        model_name = model_config['name']
        model_class = model_config['model_class']
        model_params = model_config.get('model_params', {})
        trainer_class = model_config['trainer_class']
        trainer_params = model_config.get('trainer_params', {})
        evaluator_class = model_config['evaluator_class']

        n_splits = len(cv_fold_indices)
        print(f"\n--- Starting {n_splits}-Fold Cross-Validation for PyTorch Model: {model_name} ---")

        all_fold_metrics = defaultdict(list)

        model_save_dir_for_cv = Path(model_save_base_dir) / model_name.replace(" ", "_").lower()
        os.makedirs(model_save_dir_for_cv, exist_ok=True)

        model_report_dir = Path(report_output_dir) / f"{model_name.replace(' ', '_').lower()}_cv_losses"
        os.makedirs(model_report_dir, exist_ok=True)

        # Get training parameters from trainer_params
        num_epochs_fold = trainer_params.get('num_epochs', num_epochs)
        batch_size_fold = trainer_params.get('batch_size', batch_size)
        learning_rate_fold = trainer_params.get('learning_rate', 0.001)
        criterion_name_fold = trainer_params.get('criterion_name', 'MSELoss')
        optimizer_name_fold = trainer_params.get('optimizer_name', 'Adam')

        for fold, (train_index, test_index) in enumerate(cv_fold_indices):
            print(f"\n Starting Fold {fold + 1}/{n_splits} for {model_name}")

            X_train_fold_scaled = X_cv_scaled[train_index]
            y_train_fold_scaled = y_cv_scaled[train_index]
            X_test_fold_scaled = X_cv_scaled[test_index]
            y_test_fold_scaled = y_cv_scaled[test_index]

            train_region_dist = X_cv_original_df.iloc[train_index]['operating_region'].value_counts(normalize=True)
            test_region_dist = X_cv_original_df.iloc[test_index]['operating_region'].value_counts(normalize=True)
            print(f"Fold {fold + 1} Training Region Distribution:\n{train_region_dist.round(3)}")
            print(f"Fold {fold + 1} Testing Region Distribution:\n{test_region_dist.round(3)}")

            model_instance = model_class(**model_params).to(self.device)
            fold_model_path = model_save_dir_for_cv / f'model_fold_{fold + 1}.pth'

            criterion_instance = getattr(nn, criterion_name_fold)()
            optimizer_instance = getattr(torch.optim, optimizer_name_fold)(model_instance.parameters(),
                                                                           lr=learning_rate_fold)

            trainer = trainer_class(
                model=model_instance,
                device=self.device,
                criterion=criterion_instance,
                optimizer=optimizer_instance,
                model_save_path=fold_model_path
            )

            train_losses = []
            test_losses = []

            if skip_if_exists and fold_model_path.exists():
                print(f"  Skipping training for Fold {fold + 1}: Model already exists at {fold_model_path}")
                trainer.load_model()
            else:
                X_train_tensor = torch.from_numpy(X_train_fold_scaled)
                y_train_tensor = torch.from_numpy(y_train_fold_scaled)
                X_test_tensor = torch.from_numpy(X_test_fold_scaled)
                y_test_tensor = torch.from_numpy(y_test_fold_scaled)

                train_losses, test_losses = trainer.train(
                    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
                    num_epochs=num_epochs_fold, batch_size=batch_size_fold
                )

            evaluator = evaluator_class(model_instance, self.device, self.scaler_x, self.scaler_y)

            metrics = evaluator.evaluate_model(X_test_fold_scaled, y_test_fold_scaled, X_test_original_df=X_cv_original_df.iloc[test_index])

            for key, value in metrics.items():
                if np.isfinite(value):
                    all_fold_metrics[key].append(value)
                else:
                    print(
                        f"Warning: Non-finite value encountered for metric '{key}' in Fold {fold + 1}. Skipping this value.")

            if train_losses:
                trainer.plot_losses(
                    train_losses, test_losses, num_epochs_fold,
                    model_name=model_name, fold_num=fold + 1,
                    output_dir=model_report_dir
                )

        print(f"\nCross-Validation Complete for {model_name} ({n_splits} folds)")

        # Detailed results DataFrame
        detailed_results_df = pd.DataFrame(all_fold_metrics)
        detailed_results_df['Model'] = model_name

        # Calculate averages and standard deviations from all_fold_metrics
        # This is where the summary metrics are correctly calculated
        avg_metrics = {metric: np.mean(values) for metric, values in all_fold_metrics.items()}
        std_metrics = {metric: np.std(values) for metric, values in all_fold_metrics.items()}

        # Construct summary_results_df using the correctly calculated avg_metrics/std_metrics
        summary_results_df = pd.DataFrame({
            'Model': [model_name],
            'R2_log_Avg': [avg_metrics.get('r2_log', np.nan)],
            'R2_log_Std': [std_metrics.get('r2_log', np.nan)],
            'MAE_Orig_Avg': [avg_metrics.get('mae_orig', np.nan)],
            'MAE_Orig_Std': [std_metrics.get('mae_orig', np.nan)],
            'RMSE_Orig_Avg': [avg_metrics.get('rmse_orig', np.nan)],
            'RMSE_Orig_Std': [std_metrics.get('rmse_orig', np.nan)],
            'MAPE_Orig_Avg': [avg_metrics.get('mape_orig', np.nan)],
            'MAPE_Orig_Std': [std_metrics.get('mape_orig', np.nan)]
        })

        formatter = {
            'R2_log_Avg': '{:.4f}'.format,
            'R2_log_Std': '{:.4f}'.format,
            'MAE_Orig_Avg': '{:.4e}'.format,
            'MAE_Orig_Std': '{:.4e}'.format,
            'RMSE_Orig_Avg': '{:.4e}'.format,
            'RMSE_Orig_Std': '{:.4e}'.format,
            'MAPE_Orig_Avg': '{:.4f}'.format,
            'MAPE_Orig_Std': '{:.4f}'.format
        }

        print("\nAverage and Standard Deviation of Metrics Across Folds:")
        print(summary_results_df.to_string(index=False, formatters=formatter))
        return summary_results_df, detailed_results_df

