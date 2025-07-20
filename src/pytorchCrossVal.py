# src/pytorchCrossVal.py

import torch
import numpy as np
import pandas as pd
import os
from collections import defaultdict


class PyTorchCrossValidator:
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
            model_save_base_dir (str): Base directory to save models for each fold.
            report_output_dir (str): Directory to save plots and detailed metrics.
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

        model_save_dir_for_cv = os.path.join(model_save_base_dir, model_name.replace(" ", "_").lower())
        os.makedirs(model_save_dir_for_cv, exist_ok=True)

        model_report_dir = os.path.join(report_output_dir, model_name.replace(" ", "_").lower())
        os.makedirs(model_report_dir, exist_ok=True)

        for fold, (train_index, test_index) in enumerate(cv_fold_indices):
            print(f"\n===== Starting Fold {fold + 1}/{n_splits} for {model_name} =====")

            X_train_fold_scaled = X_cv_scaled[train_index]
            y_train_fold_scaled = y_cv_scaled[train_index]
            X_test_fold_scaled = X_cv_scaled[test_index]
            y_test_fold_scaled = y_cv_scaled[test_index]

            train_region_dist = X_cv_original_df.iloc[train_index]['operating_region'].value_counts(normalize=True)
            test_region_dist = X_cv_original_df.iloc[test_index]['operating_region'].value_counts(normalize=True)
            print(f"Fold {fold + 1} Training Region Distribution:\n{train_region_dist.round(3)}")
            print(f"Fold {fold + 1} Testing Region Distribution:\n{test_region_dist.round(3)}")

            model_instance = model_class(**model_params)
            model_instance.to(self.device)

            fold_model_path = os.path.join(model_save_dir_for_cv, f'model_fold_{fold + 1}.pth')

            train_losses = []
            test_losses = []

            # Initialize trainer for the current fold
            optimizer = torch.optim.Adam(model_instance.parameters(), **trainer_params.get('optimizer_params', {}))
            trainer = trainer_class(model_instance, self.device, self.criterion, optimizer, fold_model_path)

            if skip_if_exists and os.path.exists(fold_model_path):
                print(f"  Skipping training for Fold {fold + 1}: Model already exists at {fold_model_path}")
                model_instance.load_state_dict(torch.load(fold_model_path, map_location=self.device))
                model_instance.eval()
            else:
                X_train_tensor = torch.tensor(X_train_fold_scaled, dtype=torch.float32).to(self.device)
                y_train_tensor = torch.tensor(y_train_fold_scaled, dtype=torch.float32).to(self.device)
                X_test_tensor = torch.tensor(X_test_fold_scaled, dtype=torch.float32).to(self.device)
                y_test_tensor = torch.tensor(y_test_fold_scaled, dtype=torch.float32).to(self.device)

                train_losses, test_losses = trainer.train(
                    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
                    num_epochs=num_epochs, batch_size=batch_size
                )

            # Evaluation for the current fold (whether trained or loaded)
            evaluator = evaluator_class(model_instance, self.device, self.scaler_x, self.scaler_y,
                                        self.features_for_model)

            r2, mae_orig, rmse_orig, mape_orig = evaluator.evaluate_model(X_test_fold_scaled, y_test_fold_scaled)

            all_fold_metrics['Fold'].append(fold + 1)
            all_fold_metrics['R2_log'].append(r2)
            all_fold_metrics['MAE_original'].append(mae_orig)
            all_fold_metrics['RMSE_original'].append(rmse_orig)
            all_fold_metrics['MAPE_original'].append(mape_orig)

            # Use Trainer's method to save losses plot for this fold if training occurred
            if train_losses:  # Check if losses were actually generated (i.e., training happened)
                trainer.plot_losses(
                    train_losses, test_losses, num_epochs,
                    model_name=model_name, fold_num=fold + 1,
                    output_dir=model_report_dir
                )

        print(f"\n--- Cross-Validation Complete for {model_name} ({n_splits} folds) ---")

        detailed_results_df = pd.DataFrame(all_fold_metrics)
        detailed_results_df['Model'] = model_name

        avg_metrics = {metric: np.mean(values) for metric, values in all_fold_metrics.items() if metric != 'Fold'}
        std_metrics = {metric: np.std(values) for metric, values in all_fold_metrics.items() if metric != 'Fold'}

        summary_results_df = pd.DataFrame({
            'Model': [model_name],
            'R2_log_Avg': [avg_metrics['R2_log']],
            'R2_log_Std': [std_metrics['R2_log']],
            'MAE_Orig_Avg': [avg_metrics['MAE_original']],
            'MAE_Orig_Std': [std_metrics['MAE_original']],
            'RMSE_Orig_Avg': [avg_metrics['RMSE_original']],
            'RMSE_Orig_Std': [std_metrics['RMSE_original']],
            'MAPE_Orig_Avg': [avg_metrics['MAPE_original']],
            'MAPE_Orig_Std': [std_metrics['MAPE_original']]
        })

        # Define a custom formatter for the original scale metrics to use scientific notation
        # This will be used for both printing to console and saving to CSV
        formatter = {
            'R2_log_Avg': '{:.4f}'.format,
            'R2_log_Std': '{:.4f}'.format,
            'MAE_Orig_Avg': '{:.4e}'.format,  # Scientific notation for MAE
            'MAE_Orig_Std': '{:.4e}'.format,  # Scientific notation for MAE Std
            'RMSE_Orig_Avg': '{:.4e}'.format,  # Scientific notation for RMSE
            'RMSE_Orig_Std': '{:.4e}'.format,  # Scientific notation for RMSE Std
            'MAPE_Orig_Avg': '{:.4f}'.format,
            'MAPE_Orig_Std': '{:.4f}'.format
        }

        print("\nAverage and Standard Deviation of Metrics Across Folds:")
        # Apply the formatter when printing to console
        print(summary_results_df.to_string(index=False, formatters=formatter))

        return summary_results_df, detailed_results_df

