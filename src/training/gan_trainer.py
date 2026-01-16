#src/training/gan_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from pathlib import Path
from src.core.config import settings


class GANTrainer:
    """
    Manages the training process for a single GAN (Generator and Discriminator pair)
    for a specific operating region.
    """

    def __init__(self, generator, discriminator, device, scaler_X_gan, latent_dim, model_save_base_dir, region_name):
        """
        Initializes the GANTrainer.

        Args:
            generator (torch.nn.Module): The Generator model.
            discriminator (torch.nn.Module): The Discriminator model.
            device (torch.device): The device (CPU or CUDA) to train on.
            scaler_X_gan (sklearn.preprocessing.StandardScaler): Scaler for all GAN features (inputs + target).
            latent_dim (int): Dimension of the noise vector for the Generator.
            model_save_base_dir (Path): Base directory to save trained GAN models.
            region_name (str): The name of the operating region this GAN is trained for.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.scaler_X_gan = scaler_X_gan  # Correctly renamed the scaler
        self.latent_dim = latent_dim
        self.region_name = region_name

        # Define model save paths
        self.generator_save_path = model_save_base_dir / f'generator_{region_name.lower().replace("-", "_")}.pth'
        self.discriminator_save_path = model_save_base_dir / f'discriminator_{region_name.lower().replace("-", "_")}.pth'
        os.makedirs(model_save_base_dir, exist_ok=True)

        # Loss function: Binary Cross Entropy for adversarial loss
        self.adversarial_loss = nn.BCELoss()

        # Optimizers
        self.lr_g = settings.get('gan_params.generator_learning_rate')
        self.lr_d = settings.get('gan_params.discriminator_learning_rate')
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999))

        # Move models to preferred device
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        if settings.get('global_settings.debug_mode', False):
            print(f"DEBUG: GANTrainer initialized for region: {region_name}")
            print(f"DEBUG: Generator model on: {next(self.generator.parameters()).device}")
            print(f"DEBUG: Discriminator model on: {next(self.discriminator.parameters()).device}")

    def _generate_noise(self, num_samples):
        """Generates random noise vectors for the generator input."""
        return torch.randn(num_samples, self.latent_dim, device=self.device)

    def train_gan(self, real_data_df, num_epochs, batch_size):
        """
        Trains the GAN on the provided real data.
        """
        if real_data_df.empty:
            print(f"Skipping GAN training for '{self.region_name}' region: No real data provided.")
            return [], []

        # Prepare data for PyTorch DataLoader
        gan_training_features = settings.get('gan_input.gan_training_features')
        real_data_scaled = self.scaler_X_gan.transform(real_data_df[gan_training_features])
        real_data_scaled_tensor = torch.tensor(real_data_scaled, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(real_data_scaled_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        g_losses = []
        d_losses = []

        print(f"\n    Training GAN for {self.region_name} region    ")
        print(f"  Training on {len(real_data_df)} samples for {num_epochs} epochs with batch size {batch_size}")

        real_label = 1.0
        fake_label = 0.0

        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader):
                real_samples = data[0]
                batch_size_actual = real_samples.size(0)

                # --- Train Discriminator ---
                self.optimizer_d.zero_grad()
                label = torch.full((batch_size_actual,), real_label, dtype=torch.float32, device=self.device)
                output = self.discriminator(real_samples).view(-1)
                err_d_real = self.adversarial_loss(output, label)
                err_d_real.backward()

                noise = self._generate_noise(batch_size_actual)
                fake_samples = self.generator(noise)
                label.fill_(fake_label)
                output = self.discriminator(fake_samples.detach()).view(-1)
                err_d_fake = self.adversarial_loss(output, label)
                err_d_fake.backward()
                err_d = err_d_real + err_d_fake
                self.optimizer_d.step()

                # --- Train Generator ---
                self.optimizer_g.zero_grad()
                label.fill_(real_label)
                output = self.discriminator(fake_samples).view(-1)
                err_g = self.adversarial_loss(output, label)
                err_g.backward()
                self.optimizer_g.step()

                if settings.get('global_settings.debug_mode', False) and (i % 100 == 0 or i == len(dataloader) - 1):
                    print(f"DEBUG: [{epoch}/{num_epochs}][{i}/{len(dataloader)}] "
                          f"Loss_D: {err_d.item():.4f} Loss_G: {err_g.item():.4f}")

            g_losses.append(err_g.item())
            d_losses.append(err_d.item())

            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                print(f"  Epoch [{epoch + 1}/{num_epochs}] | D Loss: {err_d.item():.4f} | G Loss: {err_g.item():.4f}")

        self.save_models()
        print(f"GAN training for {self.region_name} complete. Models saved.")
        return g_losses, d_losses

    def generate_synthetic_data(self, num_samples):
        """
        Generates synthetic data using the trained Generator and adds the correct
        operating region label before returning.
        """
        self.generator.eval()
        with torch.no_grad():
            noise = self._generate_noise(num_samples)
            generated_data_scaled = self.generator(noise).cpu().numpy()

        # Inverse transform the entire generated data batch at once
        gan_training_features = settings.get('gan_input.gan_training_features')
        generated_data_original = self.scaler_X_gan.inverse_transform(generated_data_scaled)

        # Create a DataFrame for the generated data
        generated_df = pd.DataFrame(generated_data_original, columns=gan_training_features)

        # Add the operating region label
        generated_df['operating_region'] = self.region_name

        # Calculate id from log_Id and other derived features
        generated_df['id'] = np.power(10, generated_df['log_Id'])

        # You might need to add other derived columns if your EDA script expects them.
        # Ensure your preprocessor script handles this, not the GAN script.
        # This keeps the GAN code clean and focused on generating the core features.

        print(f"Generated {num_samples} synthetic samples for {self.region_name} region.")
        return generated_df

    def save_models(self):
        """Saves the Generator and Discriminator models."""
        torch.save(self.generator.state_dict(), self.generator_save_path)
        torch.save(self.discriminator.state_dict(), self.discriminator_save_path)
        if settings.get('global_settings.debug_mode', False):
            print(f"DEBUG: Generator saved to: {self.generator_save_path}")
            print(f"DEBUG: Discriminator saved to: {self.discriminator_save_path}")

    def load_models(self):
        """Loads the Generator and Discriminator models."""
        print(f"DEBUG: Checking for models at: {self.generator_save_path}")
        if not self.generator_save_path.exists() or not self.discriminator_save_path.exists():
            if settings.get('global_settings.debug_mode', False):
                print(
                    f"DEBUG: GAN models for {self.region_name} not found at {self.generator_save_path} or {self.discriminator_save_path}.")
            return False
        self.generator.load_state_dict(torch.load(self.generator_save_path, map_location=self.device))
        self.discriminator.load_state_dict(torch.load(self.discriminator_save_path, map_location=self.device))
        self.generator.eval()
        self.discriminator.eval()
        if settings.get('global_settings.debug_mode', False):
            print(f"DEBUG: Generator and Discriminator for {self.region_name} loaded.")
        return True