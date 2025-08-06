# src/gan_augmentation/gan_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from pathlib import Path
from src.config import settings
from src.utils.helpers import calculate_vth, classify_region
import warnings


class GANTrainer:
    """
    Manages the training process for a single GAN (Generator and Discriminator pair)
    for a specific operating region.
    """

    def __init__(self, generator, discriminator, device, scaler_X, scaler_y,
                 latent_dim, model_save_base_dir, region_name):
        """
        Initializes the GANTrainer.

        Args:
            generator (torch.nn.Module): The Generator model.
            discriminator (torch.nn.Module): The Discriminator model.
            device (torch.device): The device (CPU or CUDA) to train on.
            scaler_X (sklearn.preprocessing.StandardScaler): Scaler for input features.
            scaler_y (sklearn.preprocessing.StandardScaler): Scaler for the target (log_Id).
            latent_dim (int): Dimension of the noise vector for the Generator.
            model_save_base_dir (Path): Base directory to save trained GAN models.
            region_name (str): The name of the operating region this GAN is trained for.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
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

        # Move models to preferred device (Cuda in our case)
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

        Args:
            real_data_df (pd.DataFrame): DataFrame containing real data for this region.
                                         Expected to have features and the log_Id target.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.

        Returns:
            tuple: (generator_losses, discriminator_losses) lists over epochs.
        """
        if real_data_df.empty:
            print(f"Skipping GAN training for '{self.region_name}' region: No real data provided.")
            return [], []

        # Prepare data for PyTorch DataLoader
        features_for_model = settings.get('feature_engineering.input_features')
        target_feature = settings.get('feature_engineering.target_feature')

        #Pass DataFrame slices to scalers to preserve feature names
        #Had to add this to debug the np and pd bug
        scaled_features = self.scaler_X.transform(real_data_df[features_for_model])
        scaled_target = self.scaler_y.transform(
            real_data_df[[target_feature]])  # Pass as DataFrame with double brackets

        # Concatenate scaled features and scaled target
        real_data_scaled_tensor = torch.tensor(
            np.hstack((scaled_features, scaled_target)),
            dtype=torch.float32,
            device=self.device
        )

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

                # Prints on every 10 epochs.
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
        Generates synthetic data using the trained Generator.
        The output is inverse-transformed back to the original (unscaled) feature and Id space.

        Args:
            num_samples (int): Number of synthetic samples to generate.

        Returns:
            pd.DataFrame: A DataFrame containing the generated synthetic data in original scale,
                          including 'id' and 'operating_region' columns.
        """
        self.generator.eval()
        with torch.no_grad():
            noise = self._generate_noise(num_samples)
            generated_data_scaled = self.generator(noise).cpu().numpy()

        # Separate features and target from the generated scaled data
        generated_features_scaled = generated_data_scaled[:, :-1]
        generated_log_Id_scaled = generated_data_scaled[:, -1].reshape(-1, 1)

        # Inverse transform features and target
        #Creates a dummy df with correct column names for inverse_transform
        features_for_model = settings.get('feature_engineering.input_features')
        # Ensure the number of columns matches the scalers' expected features
        if generated_features_scaled.shape[1] != len(features_for_model):
            print(f"Warning: Number of generated features ({generated_features_scaled.shape[1]}) "
                  f"does not match features_for_model ({len(features_for_model)}). "
                  f"This might cause issues with scaler_X.inverse_transform.")
            # Fallback: If there's a mismatch, try to use the columns from scaler_X directly if available
            #TODO: Debug here. Not urgent since just an fallback logic
            if hasattr(self.scaler_X, 'feature_names_in_') and self.scaler_X.feature_names_in_ is not None:
                actual_features_for_scaler = list(self.scaler_X.feature_names_in_)
            else:
                actual_features_for_scaler = features_for_model[
                                             :generated_features_scaled.shape[1]]

        else:
            actual_features_for_scaler = features_for_model

        generated_features_original = self.scaler_X.inverse_transform(
            pd.DataFrame(generated_features_scaled, columns=actual_features_for_scaler)
        )
        generated_log_Id_original = self.scaler_y.inverse_transform(
            pd.DataFrame(generated_log_Id_scaled, columns=[settings.get('feature_engineering.target_feature')])
        )

        # Convert log_Id back to original
        generated_Id_original = np.power(10, generated_log_Id_original)

        # Create a DataFrame for the generated data
        generated_df = pd.DataFrame(generated_features_original, columns=actual_features_for_scaler)
        generated_df['id'] = generated_Id_original.flatten()

        # Add the 'operating_region' column
        generated_df['operating_region'] = self.region_name

        # Ensures 'wOverL' is present if needed for later use, though GAN should generate it
        if 'w' in generated_df.columns and 'l' in generated_df.columns and 'wOverL' not in generated_df.columns:
            generated_df['wOverL'] = generated_df['w'] / generated_df['l']

        # Add other derived columns
        # These might not be directly generated by GAN but are needed for downstream processing
        # TODO: Check if w/l is correctly produced by the GANs. If not it should be removed from GAN inputs and moved to here
        if 'vg' in generated_df.columns and 'vs' in generated_df.columns and 'vgs' not in generated_df.columns:
            generated_df['vgs'] = generated_df['vg'] - generated_df['vs']
        if 'vd' in generated_df.columns and 'vs' in generated_df.columns and 'vds' not in generated_df.columns:
            generated_df['vds'] = generated_df['vd'] - generated_df['vs']
        if 'vb' in generated_df.columns and 'vs' in generated_df.columns and 'vbs' not in generated_df.columns:
            generated_df['vbs'] = generated_df['vb'] - generated_df['vs']
        if 'vs' in generated_df.columns and 'vb' in generated_df.columns and 'vsb' not in generated_df.columns:
            generated_df['vsb'] = generated_df['vs'] - generated_df['vb']

        # Recalculate Vth and operating region for generated data to ensure consistency
        # This is important as GAN might generate values that shift region distribution
        if 'vsb' in generated_df.columns:
            vth_params = settings.get('vth_params')
            # Ensure vth_params are not None
            if vth_params and all(k in vth_params for k in ['vth0', 'gamma', 'phi_f']):
                generated_df['vth'] = generated_df['vsb'].apply(
                    lambda x: calculate_vth(x, vth0=vth_params['vth0'], gamma=vth_params['gamma'],
                                            phi_f=vth_params['phi_f']))
                generated_df['operating_region'] = generated_df.apply(
                    lambda row: classify_region(row, vth_approx_val=row['vth']), axis=1)
            else:
                print(
                    "Warning: Vth parameters missing or incomplete in settings. Skipping Vth and operating region recalculation for synthetic data.")

        # Ensure 'id' is clipped to prevent issues if GAN generates negative or small values
        min_current_threshold = 1e-12
        generated_df['id'] = np.clip(generated_df['id'], a_min=min_current_threshold, a_max=None)

        # Re-calculate log_Id based on the clipped 'id'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            generated_df['log_Id'] = np.log10(generated_df['id'])

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

