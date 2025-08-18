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
            data_dim (int): Dimension of the output data (number of features).
        """
        super(Generator, self).__init__()

        # Define the main body of the network
        self.main_body = nn.Sequential(
            # Layer 1: From latent_dim to a larger hidden dimension
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),  # BatchNorm helps stabilize training
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU to prevent "dying" neurons

            # Layer 2: Further increasing complexity
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3:
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output layer that maps to the data dimension
        # We'll handle activations in the forward method
        self.output_layer = nn.Linear(512, data_dim)

    def forward(self, z):
        """
        Forward pass for the Generator.

        Args:
            z (torch.Tensor): A batch of latent vectors (noise).

        Returns:
            torch.Tensor: Generated synthetic data samples.
        """
        # Pass the latent vector through the main network body
        h = self.main_body(z)

        # Get the raw output from the final linear layer
        raw_output = self.output_layer(h)

        # IMPORTANT: Apply activation functions to enforce physical constraints.
        # w and l must be positive, and log_Id must be positive (since Id must be positive).
        # We assume the output tensor is ordered: [w, l, vgs, vds, id]
        # We can use a Softplus or ReLU activation to enforce positivity.

        # Softplus is a smooth approximation of ReLU and is often better for GANs
        # because it avoids the zero-gradient problem of ReLU.

        # Split the output tensor into parts for different activation functions
        w_output = torch.nn.functional.softplus(raw_output[:, 0].unsqueeze(1))
        l_output = torch.nn.functional.softplus(raw_output[:, 1].unsqueeze(1))
        vgs_output = raw_output[:, 2].unsqueeze(1)  # Can be positive or negative
        vds_output = raw_output[:, 3].unsqueeze(1)  # Can be positive or negative
        id_output = torch.nn.functional.softplus(raw_output[:, 4].unsqueeze(1))  # Id must be positive

        # Concatenate the outputs back into a single tensor
        generated_data = torch.cat([w_output, l_output, vgs_output, vds_output, id_output], dim=1)

        return generated_data


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
            data_dim (int): Dimension of the input data (number of features).
        """
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Layer 1: From data_dim to a hidden dimension
            nn.Linear(data_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Dropout

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
            nn.Sigmoid()  # Sigmoid to output a probability between 0 and 1
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

