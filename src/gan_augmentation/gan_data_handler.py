import pandas as pd
from src.config import settings
from pathlib import Path
from src.data_processing.preprocessor import DataPreprocessor


class GANDataHandler:
    """
    Handles data preparation for GAN training, including segregation by operating region
    and later, combining real and synthetic data.
    """

    def __init__(self):
        """
        Initializes the GANDataHandler. It will load the main preprocessed data
        which contains the 'operating_region' column.
        """
        self.preprocessor = DataPreprocessor()
        self.processed_data_dir = settings.get('paths.processed_data_dir')
        self.full_original_df = None  # This will store the full filtered original data
        self.segregated_data = {}  # Dictionary to store dataframes for each region
        self.scalers = None  # Will store scalers from preprocessor

    def load_and_segregate_data(self, force_reprocess=False):
        """
        Loads the processed data and then segregates it into separate DataFrames
        for each operating region.

        Args:
            force_reprocess (bool): If True, forces reprocessing of the main data.

        Returns:
            dict: A dictionary where keys are operating region names (str) and values
                  are pandas DataFrames containing the data for that region.
                  Returns None if data loading fails.
        """
        print("\n    GAN Data Preparation: Loading and Segregating Data    ")

        # Load the full filtered original data from the preprocessor
        if not self.preprocessor.load_or_process_data(force_reprocess=force_reprocess):
            print("Error: Failed to load or process main data for GANs. Cannot proceed.")
            return None

        self.full_original_df = self.preprocessor.get_filtered_original_data()
        self.scalers = self.preprocessor.get_scalers()  # Store scalers for later use

        if self.full_original_df.empty:
            print("No data available after initial processing for GAN segregation. Exiting.")
            return None

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
        """Returns the segregated data dictionary."""
        if not self.segregated_data:
            raise ValueError("Data not segregated. Call load_and_segregate_data() first.")
        return self.segregated_data

    def get_scalers(self):
        """Returns the X and y scalers from the main preprocessor."""
        if self.scalers is None:
            raise ValueError("Scalers not loaded. Call load_and_segregate_data() first.")
        return self.scalers

    def get_features_for_model(self):
        """Returns the list of features used for the main model, which GANs will also generate."""
        return self.preprocessor.get_features_for_model()

    def get_target_feature(self):
        """Returns the name of the target feature."""
        return settings.get('normalization.target_feature')

    def combine_and_save_augmented_data(self, augmented_dfs_by_region: dict, output_path: Path):
        """
        Combines original data with augmented data and saves the balanced dataset.

        Args:
            augmented_dfs_by_region (dict): Dictionary of augmented DataFrames for each region.
                                            Keys are region names, values are DataFrames of synthetic data.
            output_path (Path): The path where the final combined and balanced DataFrame will be saved.
        """
        print("\n    Combining Original and Augmented Data    ")
        combined_df_list = []

        # Add original data for all regions
        for region, df in self.segregated_data.items():
            combined_df_list.append(df)
            print(f"  - Added {len(df)} original samples for {region}.")

        # Add augmented data for regions that were augmented
        for region, aug_df in augmented_dfs_by_region.items():
            if not aug_df.empty:
                combined_df_list.append(aug_df)
                print(f"  - Added {len(aug_df)} synthetic samples for {region}.")
            else:
                print(f"  - No synthetic data generated for {region}.")

        final_balanced_df = pd.concat(combined_df_list, ignore_index=True)

        # Shuffle the final dataset to mix real and synthetic data
        final_balanced_df = final_balanced_df.sample(frac=1, random_state=settings.get(
            'global_settings.random_state')).reset_index(drop=True)

        # Save the combined, balanced dataset
        final_balanced_df.to_pickle(output_path)
        print(f"Combined and balanced dataset saved to: {output_path}")
        print(f"Final balanced dataset size: {len(final_balanced_df)} samples.")
        print("Final balanced region distribution:")
        print(final_balanced_df['operating_region'].value_counts(normalize=True).round(3))





