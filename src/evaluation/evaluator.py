# src/evaluation/evaluator.py
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings

class NNEvaluator:
    def __init__(self, model, device, scaler_X, scaler_y):
        self.model = model
        self.device = device
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.model.eval()

    def evaluate_model(self, X_test_scaled, y_test_scaled, X_test_original_df=None):
        # Convert inputs to Tensor if they aren't already
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

        y_test_np_scaled_log = y_test_tensor.cpu().numpy()
        y_pred_np_scaled_log = y_pred_tensor.cpu().numpy()

        # 1. Log Scale Metrics
        r2_log = r2_score(y_test_np_scaled_log, y_pred_np_scaled_log)
        mae_log = mean_absolute_error(y_test_np_scaled_log, y_pred_np_scaled_log)
        rmse_log = np.sqrt(mean_squared_error(y_test_np_scaled_log, y_pred_np_scaled_log))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mape_log = mean_absolute_percentage_error(y_test_np_scaled_log, y_pred_np_scaled_log) * 100

        metrics = {
            'R2_log': r2_log,
            'MAE_log': mae_log,
            'RMSE_log': rmse_log,
            'MAPE_log': mape_log,
        }

        # 2. Original Scale Metrics (Inverse Transform)
        if X_test_original_df is not None and 'id' in X_test_original_df.columns:
            y_test_original_id = X_test_original_df['id'].values.flatten()
            
            # Inverse transform predictions
            y_pred_orig_log = self.scaler_y.inverse_transform(y_pred_np_scaled_log.reshape(-1, 1)).flatten()
            y_pred_original_id = np.power(10, y_pred_orig_log)

            # Clip to avoid massive errors on near-zero values
            epsilon = 1e-18
            y_test_clipped = np.clip(y_test_original_id, epsilon, None)
            y_pred_clipped = np.clip(y_pred_original_id, epsilon, None)

            mae_orig = mean_absolute_error(y_test_original_id, y_pred_original_id)
            rmse_orig = np.sqrt(mean_squared_error(y_test_original_id, y_pred_original_id))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mape_orig = mean_absolute_percentage_error(y_test_clipped, y_pred_clipped) * 100

            metrics.update({
                'MAE_Orig': mae_orig,
                'RMSE_Orig': rmse_orig,
                'MAPE_Orig': mape_orig,
            })
        else:
            metrics.update({'MAE_Orig': np.nan, 'RMSE_Orig': np.nan, 'MAPE_Orig': np.nan})

        return metrics