# src/training/cv_runner.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from pathlib import Path
from src.evaluation.evaluator import NNEvaluator

class CrossValidator:
    def __init__(self, device, criterion, scaler_x, scaler_y, features_for_model):
        self.device = device
        self.criterion = criterion
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.features_for_model = features_for_model

    def run_cv(self, model_class, trainer_class, X_cv_scaled, y_cv_scaled, X_cv_original_df, cv_fold_indices,
               model_params, training_params, model_save_base_dir, report_output_dir):
        """
        Runs K-fold cross-validation.
        """
        model_name = training_params.get('model_name', 'Model')
        n_splits = len(cv_fold_indices)
        print(f"\n--- Starting {n_splits}-Fold CV for: {model_name} ---")

        all_fold_metrics = defaultdict(list)
        model_save_dir = Path(model_save_base_dir) / model_name.lower().replace(" ", "_")
        os.makedirs(model_save_dir, exist_ok=True)

        for fold, (train_idx, test_idx) in enumerate(cv_fold_indices):
            print(f"\n>>> Fold {fold + 1}/{n_splits}")

            # 1. Prepare Fold Data
            X_train = torch.tensor(X_cv_scaled[train_idx], dtype=torch.float32)
            y_train = torch.tensor(y_cv_scaled[train_idx], dtype=torch.float32)
            X_test  = torch.tensor(X_cv_scaled[test_idx], dtype=torch.float32)
            y_test  = torch.tensor(y_cv_scaled[test_idx], dtype=torch.float32)

            # 2. Instantiate Model & Optimizer per fold (Fresh start)
            model = model_class(**model_params).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])
            
            # 3. Instantiate Trainer
            fold_save_path = model_save_dir / f'model_fold_{fold + 1}.pth'
            trainer = trainer_class(
                model=model,
                device=self.device,
                criterion=self.criterion,
                optimizer=optimizer,
                model_save_path=fold_save_path
            )

            # 4. Train
            train_losses, test_losses = trainer.train(
                X_train, y_train, X_test, y_test,
                num_epochs=training_params['num_epochs'],
                batch_size=training_params['batch_size']
            )

            # 5. Evaluate
            evaluator = NNEvaluator(model, self.device, self.scaler_x, self.scaler_y)
            # Pass numpy version for evaluation to be safe with dataframe indices
            metrics = evaluator.evaluate_model(
                X_cv_scaled[test_idx], 
                y_cv_scaled[test_idx], 
                X_test_original_df=X_cv_original_df.iloc[test_idx]
            )

            for k, v in metrics.items():
                all_fold_metrics[k].append(v)

        # 6. Aggregate Results
        summary_df = pd.DataFrame(all_fold_metrics).mean().to_frame(name='Average').T
        print("\nCV Results Summary:")
        print(summary_df)
        
        summary_path = Path(report_output_dir) / f"{model_name}_cv_summary.csv"
        summary_df.to_csv(summary_path)
        print(f"Saved summary to {summary_path}")