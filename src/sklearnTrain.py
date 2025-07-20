# src/sklearnTrainer.py

import os


class SKLearnTrainer:
    def __init__(self, model, model_save_path):
        self.model = model
        self.model_save_path = model_save_path
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    #TODO: Should we remove the x_test and y_test parameters, check documentation
    def train(self, x_train, y_train, x_test=None, y_test=None):  # x_test, y_test are optional for sklearn
        """
        Trains the Scikit-learn model.
        Args:
            x_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target.
            x_test, y_test: Not used for training, but kept for consistent signature.
        Returns:
            tuple: Empty lists for train_losses, test_losses (for consistent interface).
        """
        print(f"Training {self.model.name} model...")
        self.model.fit(x_train, y_train)
        print(f"{self.model.name} training complete.")
        self.save_model()
        return [], []  # Return empty lists for consistency with NNTrainer

    def save_model(self):
        """Saves the trained Scikit-learn model."""
        self.model.save(self.model_save_path)  # Use the save method from BaseModelWrapper
        print(f"Model saved to: {self.model_save_path}")

