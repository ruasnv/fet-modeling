# src/sklearnModels.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import joblib

#TODO: Check this from SkLearn Documentation?
class BaseModelWrapper:
    """Base class for consistent model interface."""

    def __init__(self, model_instance):
        self.model = model_instance
        self.name = type(model_instance).__name__

    def fit(self, x, y):
        # Ensure y is 1D for sklearn models if it's (n, 1)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        self.model.fit(x, y)

    def predict(self, x):
        # Ensure predictions are (n, 1)
        predictions = self.model.predict(x)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

#RandomForest Regressor
class RandomForestModel(BaseModelWrapper):
    def __init__(self, random_state=42, **kwargs):
        super().__init__(RandomForestRegressor(random_state=random_state, **kwargs))
        self.name = "RandomForestRegressor"

#Ridge Regressor
class RidgeModel(BaseModelWrapper):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(Ridge(alpha=alpha, **kwargs))
        self.name = "RidgeRegressor"

#K-neighbors Regressor - Lazy Learner
class KNeighborsModel(BaseModelWrapper):
    def __init__(self, n_neighbors=5, **kwargs):
        super().__init__(KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs))
        self.name = "KNeighborsRegressor"

#XGBoostModel
class XGBoostModel(BaseModelWrapper):
    def __init__(self, random_state=42, **kwargs):
        super().__init__(XGBRegressor(random_state=random_state, **kwargs))
        self.name = "XGBoostRegressor"

