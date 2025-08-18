# scripts/run_gan_augmentation.py

import torch
import os
import sys
from pathlib import Path
import pandas as pd

# Append the parent directory to the system path to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gan_augmentation.gan_data_handler import GANDataHandler
from src.gan_augmentation.gan_model import Generator, Discriminator
from src.gan_augmentation.gan_trainer import GANTrainer
from src.utils.helpers import setup_environment
from src.config import settings


def run_gan_augmentation(force_reprocess_data=False, force_retrain_gans=False):
    """
    Orchestrates the GAN-based data augmentation pipeline.
    1. Loads and segregates data by operating region.
    2. Trains (or loads) a GAN for each region.
    3. Generates synthetic data to balance minority regions.
    4. Combines real and synthetic data into a new augmented dataset.

    Args:
        force_reprocess_data (bool): If True, forces reprocessing of the main data.
        force_retrain_gans (bool): If True, forces retraining of GANs even if models exist.
    """
    print("    Starting GAN Data Augmentation Script    ")
    setup_environment()

    #--- Load and Segregate Data ---
    gan_data_handler = GANDataHandler()
    segregated_data_dfs = gan_data_handler.load_and_segregate_data(force_reprocess=force_reprocess_data)

    if segregated_data_dfs is None:
        print("GAN augmentation aborted: Failed to load and segregate data.")
        return

    scaler_X_gan, _ = gan_data_handler.get_scalers()  # Retrieve the scaler and ignore the None value

    gan_features = settings.get('gan_input.gan_training_features')
    data_dim = len(gan_features)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for GANs: {device}")

    # ---Train (or Load) GANs for Each Region ---
    latent_dim = settings.get('gan_params.latent_dim')
    num_gan_epochs = settings.get('gan_params.num_gan_epochs')
    gan_batch_size = settings.get('gan_params.gan_batch_size')
    additional_majority_samples = settings.get('gan_params.additional_majority_samples', 0)
    gan_model_save_dir = Path(settings.get('paths.trained_model_dir')) / 'gans'
    os.makedirs(gan_model_save_dir, exist_ok=True)

    trained_gan_trainers = {}
    augmented_dfs_by_region = {}

    region_sizes = {region: len(df) for region, df in segregated_data_dfs.items()}
    if not region_sizes:
        print("No data in any region to augment. Exiting GAN augmentation.")
        return

    target_region_size = max(region_sizes.values()) + additional_majority_samples
    print(f"\nTarget size for all regions: {target_region_size} samples.")

    for region_name, df_region in segregated_data_dfs.items():
        print(f"\n    Processing GAN for {region_name} region    ")

        if df_region.empty:
            print(f"Skipping GAN for {region_name}: No data available.")
            augmented_dfs_by_region[region_name] = pd.DataFrame(columns=df_region.columns)
            continue

        generator = Generator(latent_dim=latent_dim, data_dim=data_dim)
        discriminator = Discriminator(data_dim=data_dim)

        gan_trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            device=device,
            scaler_X_gan=scaler_X_gan,
            latent_dim=latent_dim,
            model_save_base_dir=gan_model_save_dir,
            region_name=region_name
        )

        # Check if GAN models should be loaded or trained
        if settings.get('run_flags.skip_training_if_exists') and gan_trainer.load_models() and not force_retrain_gans:
            print(f"  GAN models for {region_name} loaded from disk.")
        else:
            print(f"  Training GAN models for {region_name} region...")
            gan_trainer.train_gan(
                real_data_df=df_region,
                num_epochs=num_gan_epochs,
                batch_size=gan_batch_size
            )
            print(f"  GAN training for {region_name} complete.")

        trained_gan_trainers[region_name] = gan_trainer

        # --- Data Augmentation and Balancing ---
        current_region_size = len(df_region)
        num_samples_to_generate = 0

        # Generate samples if current size is less than the target size
        if current_region_size < target_region_size:
            num_samples_to_generate = target_region_size - current_region_size
            print(f"  Generating {num_samples_to_generate} synthetic samples for {region_name}.")
            synthetic_df = gan_trainer.generate_synthetic_data(num_samples_to_generate)
            augmented_dfs_by_region[region_name] = synthetic_df
        else:
            print(f"  {region_name} already meets or exceeds target size. No synthetic data generated for this region.")
            augmented_dfs_by_region[region_name] = pd.DataFrame(
                columns=df_region.columns)  # Empty DF for synthetic data

    # --- Combine and Save Augmented Data ---
    augmented_data_output_path = Path(settings.get('paths.aug_data_dir')) / 'augmented_data.pkl'
    os.makedirs(augmented_data_output_path.parent, exist_ok=True)
    gan_data_handler.combine_and_save_augmented_data(augmented_dfs_by_region, augmented_data_output_path)

    print("   GAN Data Augmentation Script Finished   ")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GAN-based data augmentation pipeline.")
    parser.add_argument('--force_reprocess_data', action='store_true',
                        help='Force reprocessing of main data even if processed files exist.')
    parser.add_argument('--force_retrain_gans', action='store_true',
                        help='Force retraining of GANs even if models exist.')
    args = parser.parse_args()
    run_gan_augmentation(force_reprocess_data=args.force_reprocess_data,
                         force_retrain_gans=args.force_retrain_gans)

