# src/evaluation/evaluator.py

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings  # Import warnings to manage division by zero in MAPE


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

    def evaluate_model(self, X_test_scaled, y_test_scaled):
        """
        Evaluates the model on the provided test set.
        Calculates and prints metrics on both the scaled (log_Id) and original (Id) scales.

        Args:
            X_test_scaled (np.ndarray): Scaled test features (input to the model).
            y_test_scaled (np.ndarray): Scaled test target (log_Id, what the model predicts).

        Returns:
            dict: A dictionary containing various metrics:
                  - 'r2_log': R2 score on the log_Id scale.
                  - 'mae_log': Mean Absolute Error on the log_Id scale.
                  - 'rmse_log': Root Mean Squared Error on the log_Id scale.
                  - 'mae_orig': Mean Absolute Error on the original Id scale.
                  - 'rmse_orig': Root Mean Squared Error on the original Id scale.
                  - 'mape_orig': Mean Absolute Percentage Error on the original Id scale.
        """
        self.model.eval()  # Set model to evaluation mode to disable dropout, batchnorm updates, etc.

        # Convert numpy arrays to torch tensors and move to the specified device
        # Ensure float32 dtype for consistency with PyTorch models
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(self.device)

        # Perform inference without tracking gradients
        with torch.no_grad():
            y_pred_tensor = self.model(X_test_tensor)

        # Move tensors back to CPU and convert to numpy for metric calculations
        y_test_np_scaled_log = y_test_tensor.cpu().numpy()
        y_pred_np_scaled_log = y_pred_tensor.cpu().numpy()

        # Inverse transform predictions and true values back to original Id scale
        # np.power(10, ...) reverses the log10 transformation
        # .reshape(-1, 1) is crucial for StandardScaler.inverse_transform to work correctly with 1D arrays
        # .flatten() converts the (N, 1) array back to 1D for metric functions
        y_test_original_id = np.power(10,
                                      self.scaler_y.inverse_transform(y_test_np_scaled_log.reshape(-1, 1))).flatten()
        y_pred_original_id = np.power(10,
                                      self.scaler_y.inverse_transform(y_pred_np_scaled_log.reshape(-1, 1))).flatten()

        # --- Metrics on the log_Id scale (what the model directly optimizes) ---
        r2_log = r2_score(y_test_np_scaled_log, y_pred_np_scaled_log)
        mae_log = mean_absolute_error(y_test_np_scaled_log, y_pred_np_scaled_log)
        rmse_log = np.sqrt(mean_squared_error(y_test_np_scaled_log, y_pred_np_scaled_log))

        # --- Metrics on the original Id scale ---
        # Handle potential division by zero warnings for MAPE, especially if y_test_original_id can contain zeros.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # Ignore warnings like divide by zero
            mape_original_id = mean_absolute_percentage_error(y_test_original_id, y_pred_original_id) * 100

        mae_original_id = mean_absolute_error(y_test_original_id, y_pred_original_id)
        rmse_original_id = np.sqrt(mean_squared_error(y_test_original_id, y_pred_original_id))

        # Print evaluation results
        print(f"\nEvaluating Model: {self.model.__class__.__name__}")
        print(f"  R-squared (R2) [log_Id scale]: {r2_log:.4f}")
        print(f"  Mean Absolute Error (MAE) [log_Id scale]: {mae_log:.4f}")
        print(f"  Root Mean Squared Error (RMSE) [log_Id scale]: {rmse_log:.4f}")
        print(f"  Mean Absolute Error (MAE) [Original Id scale]: {mae_original_id:.4e}")
        print(f"  Root Mean Squared Error (RMSE) [Original Id scale]: {rmse_original_id:.4e}")
        # Note: MAPE can be very sensitive to small true values. Handle with care if your Id values approach zero.
        print(f"  Mean Absolute Percentage Error (MAPE) [Original Id scale]: {mape_original_id:.2f}%")

        # Return metrics in a dictionary for easy access and logging
        return {
            'r2_log': r2_log,
            'mae_log': mae_log,
            'rmse_log': rmse_log,
            'mae_orig': mae_original_id,
            'rmse_orig': rmse_original_id,
            'mape_orig': mape_original_id
        }