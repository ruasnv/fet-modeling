# src/pytorchTrain.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt

class NNTrainer:
    def __init__(self, model, device, criterion, optimizer, model_save_path):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_save_path = model_save_path
        self.best_test_loss = float('inf')
        model_dir = os.path.dirname(model_save_path) or '.'
        os.makedirs(model_dir, exist_ok=True)

    def train(self, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, num_epochs, batch_size):
        """
        Trains the neural network model.
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
        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        print(f"Training on {len(x_train_tensor)} samples for {num_epochs} epochs with batch size {batch_size}")

        for epoch in range(num_epochs):
            self.model.train()  # Set model to training mode
            running_train_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)

            avg_train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(avg_train_loss)

            # Evaluate on test set after each epoch
            self.model.eval()  # Set model to evaluation mode

            #TODO: If exploding gradients ever occur, use gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            #Good practise to empty the cache for tight GPU memories
            torch.cuda.empty_cache()

            # direct test loss calculation
            # Since we are not performin backward, we should use torch.no_grad()
            # "when you are sure that you will not call Tensor. backward(). It will reduce memory consumption for computations"
            with torch.no_grad():
                test_outputs = self.model(x_test_tensor.to(self.device))
                test_loss = self.criterion(test_outputs, y_test_tensor.to(self.device)).item()
            test_losses.append(test_loss)

            print(f'Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.6f} | Test Loss: {test_loss:.6f}')

            # Save the model if it's the best so far on the test set
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f" Best model saved to {self.model_save_path}")

        print(f"Training complete. Best test loss: {self.best_test_loss:.6f}")
        return train_losses, test_losses

    def plot_losses(self, train_losses, test_losses, num_epochs, model_name, fold_num=None,
                    output_dir="reports/models"):
        """
        Plots training and testing losses and saves the plot.
        Args:
            train_losses (list): List of training losses per epoch.
            test_losses (list): List of testing losses per epoch.
            num_epochs (int): Total number of epochs.
            model_name (str): Name of the model for the plot title and filename.
            fold_num (int, optional): The current fold number for CV plots. Defaults to None.
            output_dir (str): Directory where the plot will be saved.
        """
        if not train_losses or not test_losses:
            print(
                f"No loss history to plot for {model_name} (Fold {fold_num if fold_num is not None else 'Final'}). Skipping loss plot.")
            return

        #Configs
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), test_losses, label='Testing Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')

        if fold_num is not None:
            title = f'{model_name} Fold {fold_num} Training and Testing Loss over Epochs'
            filename = f'{model_name.replace(" ", "_").lower()}_fold_{fold_num}_losses.png'
        else:
            title = f'{model_name} Training and Testing Loss over Epochs'
            filename = f'{model_name.replace(" ", "_").lower()}_final_losses.png'

        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Loss plot saved to: {os.path.join(output_dir, filename)}")

        #If raw loss arrays are needed in the future for future analysis use the code below.
        # import numpy as np
        #np.save(os.path.join(output_dir, f'{model_name}_train_losses.npy'), train_losses)
        #np.save(os.path.join(output_dir, f'{model_name}_test_losses.npy'), test_losses)


