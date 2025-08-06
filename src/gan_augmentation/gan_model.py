# src/gan_augmentation/gan_model.py

import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    The Generator network for the GAN.
    It takes a latent space vector (noise) as input and generates synthetic data samples.
    The output dimension matches the number of input features + 1 (for the log_Id target).
    """
    def __init__(self, latent_dim, data_dim):
        """
        Initializes the Generator.

        Args:
            latent_dim (int): Dimension of the input latent vector (noise).
            data_dim (int): Dimension of the output data (number of features + 1 for log_Id).
        """
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # Layer 1: From latent_dim to a larger hidden dimension
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128), # BatchNorm helps stabilize GAN training
            nn.LeakyReLU(0.2, inplace=True), # LeakyReLU for generator to avoid vanishing gradients

            # Layer 2: Further increasing complexity
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3:
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer: Maps to the desired data_dim
            # No activation here, as StandardScaler-normalized data can be unbounded (positive/negative)
            # If your data was strictly bounded (e.g., [0,1]), you might use nn.Sigmoid() or nn.Tanh()
            nn.Linear(512, data_dim)
        )

    def forward(self, z):
        """
        Forward pass for the Generator.

        Args:
            z (torch.Tensor): A batch of latent vectors (noise).

        Returns:
            torch.Tensor: Generated synthetic data samples.
        """
        return self.model(z)


class Discriminator(nn.Module):
    """
    The Discriminator network for the GAN.
    It takes a data sample (real or synthetic) as input and outputs a single probability
    indicating whether the sample is real (closer to 1) or fake (closer to 0).
    """
    def __init__(self, data_dim):
        """
        Initializes the Discriminator.

        Args:
            data_dim (int): Dimension of the input data (number of features + 1 for log_Id).
        """
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Layer 1: From data_dim to a hidden dimension
            nn.Linear(data_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3), # Dropout helps prevent overfitting and improves stability

            # Layer 2:
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Layer 3:
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Output layer: Single output for binary classification (real/fake)
            nn.Linear(128, 1),
            nn.Sigmoid() # Sigmoid to output a probability between 0 and 1
        )

    def forward(self, x):
        """
        Forward pass for the Discriminator.

        Args:
            x (torch.Tensor): A batch of data samples (real or synthetic).

        Returns:
            torch.Tensor: Probability that the input sample is real.
        """
        return self.model(x)

