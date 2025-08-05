# src/evaluation/evaluator.py

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings

class NNEvaluator:
    def __init__(self, model, device, scaler_X, scaler_y):
        """
        Initializes the NNEvaluator.

        Args:
            model (torch.nn.Module): The trained neural network model.
            device (torch.device): The device (CPU or CUDA) the model is on.
            scaler_X (sklearn.preprocessing.StandardScaler): Scaler fitted on input features (X).
            scaler_y (sklearn.preprocessing.StandardScaler): Scaler fitted on target feature (y, log_Id).
        """
        self.model = model
        self.device = device
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.model.eval()  # Set model to evaluation mode

    def evaluate_model(self, X_test_scaled, y_test_scaled, X_test_original_df=None):
        """
        Evaluates the model's performance using R2, MAE, RMSE on both scaled and original data,
        and MAPE on original data.

        Args:
            X_test_scaled (np.ndarray or torch.Tensor): Scaled test features (input to the model).
            y_test_scaled (np.ndarray or torch.Tensor): Scaled test target (log_Id, what the model predicts).
            X_test_original_df (pd.DataFrame, optional): Original (unscaled) test features,
                                                         including the original 'id' column.
                                                         Required for original scale metrics.

        Returns:
            dict: A dictionary containing various evaluation metrics.
        """
        # Ensure inputs are tensors and on the correct device
        if isinstance(X_test_scaled, np.ndarray):
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
        else:
            X_test_tensor = X_test_scaled.to(self.device).float()

        if isinstance(y_test_scaled, np.ndarray):
            y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(self.device)
        else:
            y_test_tensor = y_test_scaled.to(self.device).float()

        with torch.no_grad():
            y_pred_tensor = self.model(X_test_tensor)

        # Move tensors back to CPU and convert to numpy for metric calculations
        y_test_np_scaled_log = y_test_tensor.cpu().numpy()
        y_pred_np_scaled_log = y_pred_tensor.cpu().numpy()

        # --- Metrics on the log_Id scale (what the model directly optimizes) ---
        r2_log = r2_score(y_test_np_scaled_log, y_pred_np_scaled_log)
        mae_log = mean_absolute_error(y_test_np_scaled_log, y_pred_np_scaled_log)
        rmse_log = np.sqrt(mean_squared_error(y_test_np_scaled_log, y_pred_np_scaled_log))

        # Calculate MAPE on the log_Id scale
        # No clipping needed here as log_Id values should already be finite and non-zero
        # due to the clipping applied during preprocessing.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # Ignore warnings like divide by zero
            mape_log = mean_absolute_percentage_error(y_test_np_scaled_log, y_pred_np_scaled_log) * 100

        metrics = {
            'r2_log': r2_log,
            'mae_log': mae_log,
            'rmse_log': rmse_log,
            'mape_log': mape_log,  # Add MAPE on log scale
        }

        # --- Metrics on the original Id scale ---
        if X_test_original_df is not None and 'id' in X_test_original_df.columns:
            # Use the TRUE original 'id' values from the DataFrame for ground truth
            y_test_original_id = X_test_original_df['id'].values.flatten()

            # Inverse transform predictions from log_Id scale to original Id scale
            y_pred_orig_log = self.scaler_y.inverse_transform(y_pred_np_scaled_log.reshape(-1, 1)).flatten()
            y_pred_original_id = np.power(10, y_pred_orig_log)

            # Clip very small values in both true and predicted arrays for MAPE calculation
            # This prevents division by zero or extremely large percentage errors for near-zero values.
            # A small positive epsilon for true values to prevent division by zero in MAPE
            epsilon = 1e-18  # A very small current in Amperes (10^-18 A = attoampere)
            y_test_original_id_clipped = np.clip(y_test_original_id, epsilon, None)
            y_pred_original_id_clipped = np.clip(y_pred_original_id, epsilon, None)

            mae_orig = mean_absolute_error(y_test_original_id, y_pred_original_id)
            rmse_orig = np.sqrt(mean_squared_error(y_test_original_id, y_pred_original_id))

            # Calculate MAPE using the clipped values to avoid inf/NaN
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # Ignore warnings like divide by zero
                mape_orig = mean_absolute_percentage_error(y_test_original_id_clipped, y_pred_original_id_clipped) * 100

            metrics.update({
                'mae_orig': mae_orig,
                'rmse_orig': rmse_orig,
                'mape_orig': mape_orig,
            })
        else:
            print(
                "Warning: X_test_original_df not provided or 'id' column missing. Cannot calculate metrics on original scale.")
            metrics.update({
                'mae_orig': np.nan,
                'rmse_orig': np.nan,
                'mape_orig': np.nan,
            })

        # Print evaluation results
        print(f"\nEvaluating Model: {self.model.__class__.__name__}")
        print(f"  R-squared (R2) [log_Id scale]: {metrics['r2_log']:.4f}")
        print(f"  Mean Absolute Error (MAE) [log_Id scale]: {metrics['mae_log']:.4f}")
        print(f"  Root Mean Squared Error (RMSE) [log_Id scale]: {metrics['rmse_log']:.4f}")
        print(
            f"  Mean Absolute Percentage Error (MAPE) [log_Id scale]: {metrics['mape_log']:.2f}%")  # Print MAPE on log scale

        if 'mae_orig' in metrics and not np.isnan(metrics['mae_orig']):
            print(f"  Mean Absolute Error (MAE) [Original Id scale]: {metrics['mae_orig']:.4e}")
            print(f"  Root Mean Squared Error (RMSE) [Original Id scale]: {metrics['rmse_orig']:.4e}")
            print(f"  Mean Absolute Percentage Error (MAPE) [Original Id scale]: {metrics['mape_orig']:.2f}%")
        else:
            print("  Original scale metrics not available.")

        # Return metrics in a dictionary for easy access and logging
        return metrics

