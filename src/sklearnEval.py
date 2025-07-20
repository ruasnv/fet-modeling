# src/sklearnEval.py

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


class SKLearnEvaluator:
    def __init__(self, model, scaler_x, scaler_y, features_for_model):
        self.model = model
        self.scaler_X = scaler_x
        self.scaler_y = scaler_y
        self.features_for_model = features_for_model

    def evaluate_model(self, x_test_scaled, y_test_scaled):
        """
        Evaluates the Scikit-learn model on the test set and prints metrics.
        Args:
            x_test_scaled (np.ndarray): Scaled test features.
            y_test_scaled (np.ndarray): Scaled test target (log_Id).
        Returns:
            tuple: R2 (log_Id), MAE (original Id), RMSE (original Id), MAPE (original Id)
        """
        y_pred_scaled_log = self.model.predict(x_test_scaled)

        # Ensure y_test_scaled and y_pred_scaled_log are 2D for inverse_transform
        if y_test_scaled.ndim == 1:
            y_test_scaled = y_test_scaled.reshape(-1, 1)
        if y_pred_scaled_log.ndim == 1:
            y_pred_scaled_log = y_pred_scaled_log.reshape(-1, 1)

        #TODO: Duplicate, merge two logics

        # Inverse transform predictions and true values back to original Id scale
        y_test_original_id = np.power(10, self.scaler_y.inverse_transform(y_test_scaled))
        y_pred_original_id = np.power(10, self.scaler_y.inverse_transform(y_pred_scaled_log))

        # Metrics on the log_Id scale (what the model directly optimizes)
        r2_log = r2_score(y_test_scaled, y_pred_scaled_log)
        mae_log = mean_absolute_error(y_test_scaled, y_pred_scaled_log)
        rmse_log = np.sqrt(mean_squared_error(y_test_scaled, y_pred_scaled_log))

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

    #TODO: Should we make the function static?
    def plot_losses(self, train_losses, test_losses, num_epochs):
        """Plots training and testing losses. (Not applicable for most sklearn models)."""
        print("Loss plotting is typically not applicable for this type of Scikit-learn model.")

