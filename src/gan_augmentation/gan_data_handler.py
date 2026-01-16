import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import settings
from pathlib import Path
from src.data_processing.preprocessor import DataPreprocessor


class GANDataHandler:
    """
    Handles data preparation for GAN training, including segregation by operating region
    and later, combining real and synthetic data.
    """

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.processed_data_dir = settings.get('paths.processed_data_dir')
        self.full_original_df = None
        self.segregated_data = {}
        self.scaler_X_gan = None
        self.scaler_y = None
        self.gan_features = settings.get('gan_input.gan_training_features')

    def load_and_segregate_data(self, force_reprocess=False):
        print("\n    GAN Data Preparation: Loading and Segregating Data    ")

        if not self.preprocessor.load_or_process_data(force_reprocess=force_reprocess):
            print("Error: Failed to load or process main data for GANs. Cannot proceed.")
            return None

        self.full_original_df = self.preprocessor.get_filtered_original_data()

        # The scaler for the y-target is no longer needed since a single scaler for all GAN features is used.
        # _, self.scaler_y = self.preprocessor.get_scalers()

        if self.full_original_df.empty:
            print("No data available after initial processing for GAN segregation. Exiting.")
            return None

        # --- Create and fit a single scaler for all GAN features (inputs + target) ---
        print("  - Creating and fitting a new StandardScaler for all GAN features...")
        gan_data_subset = self.full_original_df[self.gan_features]
        self.scaler_X_gan = StandardScaler()
        self.scaler_X_gan.fit(gan_data_subset)
        print("  - GAN-specific StandardScaler created and fitted successfully.")

        regions = ["Cut-off", "Linear", "Saturation"]
        print(f"Total samples in full original data: {len(self.full_original_df)}")
        print("Segregating data by operating region:")

        for region in regions:
            region_df = self.full_original_df[self.full_original_df['operating_region'] == region].copy()
            self.segregated_data[region] = region_df
            print(f"  - {region} region: {len(region_df)} samples")
            if region_df.empty:
                print(f"    Warning: No data found for the '{region}' region. This GAN will not be trained.")

        print("Data segregation complete.")
        return self.segregated_data

    def get_segregated_data(self):
        if not self.segregated_data:
            raise ValueError("Data not segregated. Call load_and_segregate_data() first.")
        return self.segregated_data

    def get_scalers(self):
        """
        Returns a single scaler for all GAN features. The y_scaler is no longer needed
        as it's now part of the main GAN scaler.
        """
        if self.scaler_X_gan is None:
            raise ValueError("Scalers not loaded. Call load_and_segregate_data() first.")
        # FIX: Return a single scaler for X and None for y, as y is now included in X.
        return self.scaler_X_gan, None

    def get_features_for_model(self):
        """Returns the list of features specifically for the GAN model (inputs + target)."""
        return self.gan_features

#TODO: REDUNDANCY!
    def get_target_feature(self):
        # FIX: This method is now redundant for the GAN's training data pipeline.
        return settings.get('gan_input.target_feature')

    def combine_and_save_augmented_data(self, augmented_dfs_by_region: dict, output_path: Path):
        print("\n    Combining Original and Augmented Data    ")
        combined_df_list = []

        for region, df in self.segregated_data.items():
            combined_df_list.append(df)
            print(f"  - Added {len(df)} original samples for {region}.")

        for region, aug_df in augmented_dfs_by_region.items():
            if not aug_df.empty:
                combined_df_list.append(aug_df)
                print(f"  - Added {len(aug_df)} synthetic samples for {region}.")
            else:
                print(f"  - No synthetic data generated for {region}.")

        final_balanced_df = pd.concat(combined_df_list, ignore_index=True)
        final_balanced_df = final_balanced_df.sample(frac=1, random_state=settings.get(
            'global_settings.random_state')).reset_index(drop=True)

        final_balanced_df.to_pickle(output_path)
        print(f"Combined and balanced dataset saved to: {output_path}")
        print(f"Final balanced dataset size: {len(final_balanced_df)} samples.")
        print("Final balanced region distribution:")
        print(final_balanced_df['operating_region'].value_counts(normalize=True).round(3))