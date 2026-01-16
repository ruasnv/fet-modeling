# src/training/simple_nn_trainer.py
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader
from src.core.config import settings

class NNTrainer:
    def __init__(self, model, device, criterion, optimizer, model_save_path):
        """
        Initializes the NNTrainer.

        Args:
            model (torch.nn.Module): The neural network model to train.
            device (torch.device): The device (CPU or CUDA) to run training on.
            criterion (torch.nn.Module): The instantiated loss function (e.g., nn.MSELoss()).
            optimizer (torch.optim.Optimizer): The instantiated optimizer (e.g., torch.optim.Adam(...)).
            model_save_path (Path): Full path to save the best model's state_dict.
        """
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_save_path = Path(model_save_path)
        self.best_test_loss = float('inf')  # Tracks the best test loss for model saving
        # Ensure the directory for saving the model exists
        os.makedirs(self.model_save_path.parent, exist_ok=True)


    def train(self, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, num_epochs, batch_size):
        """
        Trains the neural network model and evaluates on a test set each epoch.

        Args:
            x_train_tensor (torch.Tensor): Training features.
            y_train_tensor (torch.Tensor): Training target.
            x_test_tensor (torch.Tensor): Test features (for evaluation during training).
            y_test_tensor (torch.Tensor): Test target (for evaluation during training).
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.

        Returns:
            tuple: Lists of training and testing losses per epoch.
        """
        train_losses = []
        test_losses = []

        # Ensure tensors are on the correct device and have correct dtype
        x_train_tensor = x_train_tensor.to(self.device).float()
        y_train_tensor = y_train_tensor.to(self.device).float()
        x_test_tensor = x_test_tensor.to(self.device).float()
        y_test_tensor = y_test_tensor.to(self.device).float()

        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        print(f"  Training on {len(x_train_tensor)} samples for {num_epochs} epochs with batch size {batch_size}")

        for epoch in range(num_epochs):
            self.model.train()  # Set model to training mode
            running_train_loss = 0.0

            for inputs, targets in train_loader:
                # inputs and targets are already on device from TensorDataset and .to(self.device) above!
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)

            avg_train_loss = running_train_loss / len(x_train_tensor)
            train_losses.append(avg_train_loss)

            # Evaluate on test set after each epoch
            self.model.eval()  # Set model to evaluation mode
            # Good practice to empty the cache for tight GPU memories
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Using torch.no_grad() is good for evaluation to save memory and avoid gradient calculations
            with torch.no_grad():
                test_outputs = self.model(x_test_tensor)
                test_loss = self.criterion(test_outputs, y_test_tensor).item()
            test_losses.append(test_loss)

            print(f'  Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.6f} | Test Loss: {test_loss:.6f}')

            # Save the model if it's the best so far on the test set
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"  Best model saved to {self.model_save_path}")

        print(f"Training complete. Best test loss: {self.best_test_loss:.6f}")
        return train_losses, test_losses

    def save_model(self):
        """Saves the trained model to the specified path."""
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

    def load_model(self):
        """Loads a model from the specified path."""
        if self.model_save_path.exists():
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {self.model_save_path}")
            return True
        else:
            print(f"Model not found at {self.model_save_path}")
            return False

    def plot_losses(self, train_losses, test_losses, num_epochs, model_name, fold_num=None,
                    output_dir=None):  # output_dir is now explicitly passed and required
        """
        Plots training and testing losses and saves the plot.

        Args:
            train_losses (list): List of training losses per epoch.
            test_losses (list): List of testing losses per epoch.
            num_epochs (int): Total number of epochs.
            model_name (str): Name of the model for the plot title and filename.
            fold_num (int, optional): The current fold number for CV plots. Defaults to None (for final model).
            output_dir (Path): Directory where the plot will be saved. Must be provided.
        """
        if not train_losses or not test_losses:
            print(
                f"No loss history to plot for {model_name} (Fold {fold_num if fold_num is not None else 'Final'}). Skipping loss plot.")
            return
        if output_dir is None:
            raise ValueError("output_dir must be provided to plot_losses.")

        print(f"  Plotting Loss for {model_name}, (Fold {fold_num if fold_num is not None else 'Final Model'})")

        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=settings.get('global_settings.figure_figsize'))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), test_losses, label='Testing Loss')
        plt.xlabel('Epoch', fontsize=settings.get('global_settings.axes_labelsize'))
        plt.ylabel('Loss (MSE)', fontsize=settings.get('global_settings.axes_labelsize'))

        if fold_num is not None:
            title = f'{model_name} Fold {fold_num} Training and Testing Loss over Epochs'
            filename = f'{model_name.replace(" ", "_").lower()}_fold_{fold_num}_losses.png'
        else:
            title = f'{model_name} Training and Testing Loss over Epochs (Final Model)'
            filename = f'{model_name.replace(" ", "_").lower()}_final_losses.png'

        plt.title(title, fontsize=settings.get('global_settings.axes_titlesize'))
        plt.legend(fontsize=settings.get('global_settings.legend_fontsize'))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_filepath = Path(output_dir) / filename
        plt.savefig(plot_filepath)
        plt.close()
        print(f"  Loss plot saved to: {plot_filepath}")