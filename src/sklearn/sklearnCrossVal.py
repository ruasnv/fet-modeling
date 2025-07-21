# src/sklearn_cross_validator.py

import numpy as np
import pandas as pd
import os
from collections import defaultdict


class SKLearnCrossValidator:
    def __init__(self, scaler_x, scaler_y, features_for_model):
        self.scaler_X = scaler_x
        self.scaler_y = scaler_y
        self.features_for_model = features_for_model

    def run_cv(self, model_config, x_cv_scaled, y_cv_scaled, x_cv_original_df, cv_fold_indices,
               model_save_base_dir="trained_models/k_fold_models",
               skip_if_exists=True):
        """
        Runs K-fold cross-validation for a Scikit-learn model.

        Args:
            model_config (dict): Configuration for the model, trainer, and evaluator.
            x_cv_scaled (np.ndarray): Full CV pool features, scaled.
            y_cv_scaled (np.ndarray): Full CV pool target, scaled (log_Id).
            x_cv_original_df (pd.DataFrame): Original DataFrame for stratification verification.
            cv_fold_indices (list): List of (train_idx, test_idx) tuples for each fold.
            model_save_base_dir (str): Base directory to save models for each fold.
            skip_if_exists (bool): If True, skips training if a saved model for the fold exists.

        Returns:
            pd.DataFrame: A DataFrame containing average and std deviation of metrics across folds.
            list: Empty list (for consistency with PyTorch CV, as sklearn models don't have epoch losses).
            list: Empty list (for consistency with PyTorch CV, as sklearn models don't have epoch losses).
        """
        model_name = model_config['name']
        model_class = model_config['model_class']
        model_params = model_config.get('model_params', {})
        trainer_class = model_config['trainer_class']
        evaluator_class = model_config['evaluator_class']

        n_splits = len(cv_fold_indices)
        print(f"\n--- Starting {n_splits}-Fold Cross-Validation for Scikit-learn Model: {model_name} ---")

        all_fold_metrics = defaultdict(list)

        model_save_dir_for_cv = os.path.join(model_save_base_dir, model_name.replace(" ", "_").lower())
        os.makedirs(model_save_dir_for_cv, exist_ok=True)

        for fold, (train_index, test_index) in enumerate(cv_fold_indices):
            print(f"\n===== Starting Fold {fold + 1}/{n_splits} for {model_name} =====")

            # Prepare data for the current fold
            X_train_fold_scaled = x_cv_scaled[train_index]
            y_train_fold_scaled = y_cv_scaled[train_index]
            X_test_fold_scaled = x_cv_scaled[test_index]
            y_test_fold_scaled = y_cv_scaled[test_index]

            # Verify stratification
            train_region_dist = x_cv_original_df.iloc[train_index]['operating_region'].value_counts(normalize=True)
            test_region_dist = x_cv_original_df.iloc[test_index]['operating_region'].value_counts(normalize=True)
            print(f"Fold {fold + 1} Training Region Distribution:\n{train_region_dist.round(3)}")
            print(f"Fold {fold + 1} Testing Region Distribution:\n{test_region_dist.round(3)}")

            # Model Initialization for the current fold
            model_instance = model_class(**model_params)

            fold_model_path = os.path.join(model_save_dir_for_cv, f'model_fold_{fold + 1}.joblib')

            # --- Skip Training Logic ---
            if skip_if_exists and os.path.exists(fold_model_path):
                print(f"  Skipping training for Fold {fold + 1}: Model already exists at {fold_model_path}")
                model_instance.load(fold_model_path)  # Use BaseModelWrapper's load method
            else:
                # Proceed with training
                trainer = trainer_class(model_instance, fold_model_path)
                trainer.train(X_train_fold_scaled, y_train_fold_scaled)

            # Evaluation for the current fold (whether trained or loaded)
            evaluator = evaluator_class(model_instance, self.scaler_X, self.scaler_y, self.features_for_model)

            r2, mae_orig, rmse_orig, mape_orig = evaluator.evaluate_model(X_test_fold_scaled, y_test_fold_scaled)

            all_fold_metrics['R2_log'].append(r2)
            all_fold_metrics['MAE_original'].append(mae_orig)
            all_fold_metrics['RMSE_original'].append(rmse_orig)
            all_fold_metrics['MAPE_original'].append(mape_orig)

        # --- MOVED OUTSIDE THE FOR LOOP ---
        print(f"\n--- Cross-Validation Complete for {model_name} ({n_splits} folds) ---")

        avg_metrics = {metric: np.mean(values) for metric, values in all_fold_metrics.items()}
        std_metrics = {metric: np.std(values) for metric, values in all_fold_metrics.items()}

        results_df = pd.DataFrame({
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
        print("\nAverage and Standard Deviation of Metrics Across Folds:")
        print(results_df.to_string(index=False, float_format="%.4f"))

        return results_df, [], []  # Return empty lists for losses for consistency

