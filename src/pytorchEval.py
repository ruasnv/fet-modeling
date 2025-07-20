# src/pytorchEval.py

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class NNEvaluator:
    def __init__(self, model, device, scaler_X, scaler_y, features_for_model):
        self.model = model
        self.device = device
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.features_for_model = features_for_model

    def evaluate_model(self, X_test_scaled, y_test_scaled):
        """
        Evaluates the model on the test set and prints metrics, including MAPE.
        Args:
            X_test_scaled (np.ndarray): Scaled test features.
            y_test_scaled (np.ndarray): Scaled test target (log_Id).
        Returns:
            tuple: R2 (log_Id), MAE (original Id), RMSE (original Id), MAPE (original Id)
        """
        self.model.eval() # Set model to evaluation mode
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred_tensor = self.model(X_test_tensor)

        y_test_np_scaled_log = y_test_tensor.cpu().numpy()
        y_pred_np_scaled_log = y_pred_tensor.cpu().numpy()

        # Inverse transform predictions and true values back to original Id scale for MAPE calculation
        y_test_original_id = np.power(10, self.scaler_y.inverse_transform(y_test_np_scaled_log))
        y_pred_original_id = np.power(10, self.scaler_y.inverse_transform(y_pred_np_scaled_log))

        # Metrics on the log_Id scale (what the model directly optimizes)
        r2_log = r2_score(y_test_np_scaled_log, y_pred_np_scaled_log)
        mae_log = mean_absolute_error(y_test_np_scaled_log, y_pred_np_scaled_log)
        rmse_log = np.sqrt(mean_squared_error(y_test_np_scaled_log, y_pred_np_scaled_log))

        # Metrics on the original Id scale
        mape_original_id = mean_absolute_percentage_error(y_test_original_id, y_pred_original_id) * 100
        mae_original_id = mean_absolute_error(y_test_original_id, y_pred_original_id)
        rmse_original_id = np.sqrt(mean_squared_error(y_test_original_id, y_pred_original_id))


        print(f"\n--- Model Evaluation on Test Set ---")
        print(f"  R-squared (R2) [log_Id scale]: {r2_log:.4f}")
        print(f"  Mean Absolute Error (MAE) [log_Id scale]: {mae_log:.4f}")
        print(f"  Root Mean Squared Error (RMSE) [log_Id scale]: {rmse_log:.4f}")
        print(f"  Mean Absolute Error (MAE) [Original Id scale]: {mae_original_id:.4e}")
        print(f"  Root Mean Squared Error (RMSE) [Original Id scale]: {rmse_original_id:.4e}")
        print(f"  Mean Absolute Percentage Error (MAPE) [Original Id scale]: {mape_original_id:.2f}%")

        return r2_log, mae_original_id, rmse_original_id, mape_original_id

