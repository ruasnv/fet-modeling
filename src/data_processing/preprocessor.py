# src/data_processing/preprocessor.py
import os
import warnings
import joblib
import numpy as np
import path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from .data_loader import DataLoader # Use relative import if data_loader.py is in the same package
from src.utils.helpers import calculate_vth, classify_region
from src.config import settings

class DataPreprocessor:
    def __init__(self): # Just takes one merged config dictionary
        """
        Initializes the DataPreprocessor with the complete merged configuration data.
        """

        # Access all parameters from the single merged config
        self.raw_data_path = settings.get('raw_data_path')
        self.processed_data_dir = settings.get('processed_data_dir')

        self.df = None
        self.filtered_original_df = None
        self.scaler_X = None
        self.scaler_y = None

        self.features_for_model = settings.get('features_for_model')
        if settings.get('include_w_over_l'):
            self.features_for_model.append('wOverL')

        self.target_feature = settings.get('target_feature')
        self.stratify_column = settings.get('stratify_column')

        # Processed data components (will be set after preparation or loading)
        self.X_cv_scaled = None
        self.y_cv_scaled = None
        self.X_cv_original_df = None
        self.cv_fold_indices = []

        self.X_final_test_scaled = None
        self.y_final_test_scaled = None
        self.X_final_test_original_df = None

    # Remove _load_config as the data is passed directly
    # def _load_config(self, config_path):
    #     """Loads a YAML configuration file."""
    #     with open(config_path, 'r') as f:
    #         return yaml.safe_load(f)

    def _apply_filters(self, df):
        """Applies filters based on data_config."""
        initial_rows = len(df)
        filtered_df = df.copy()

        # Id > 0
        if settings.get('id_greater_than_zero'):
            filtered_df = filtered_df[filtered_df['id'] > 0].copy()
            print(f"  - After Id > 0 filter: {len(filtered_df)} rows")

        # Vd > 0
        if settings.get('vds_greater_than_zero'):
            filtered_df = filtered_df[filtered_df['vd'] > 0].copy()
            print(f"  - After Vd > 0 filter: {len(filtered_df)} rows")

        # Temperature filter
        temp_filter = settings.get('temperature_filter')
        if temp_filter is not None:
            filtered_df = filtered_df[filtered_df['temp'] == temp_filter].copy()
            print(f"  - After Temp = {temp_filter}C filter: {len(filtered_df)} rows")

        print(f"Total filtered rows: {len(filtered_df)} (from {initial_rows} initial rows)")
        return filtered_df

    def _perform_feature_engineering(self, df):
        """Performs feature engineering based on data_config."""
        engineered_df = df.copy()

        # Add W/L feature
        if settings.get('include_w_over_l'):
            engineered_df['wOverL'] = engineered_df['w'] / engineered_df['l']
            print("  - Added 'wOverL' feature.")

        # Log transform for Id
        if settings.get('log_transform_id'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # Ignore log of zero/negative warnings
                engineered_df['log_Id'] = np.log10(engineered_df['id'])
            print("  - Applied log10 transformation to 'id' to create 'log_Id'.")

        # Calculate Vth and Operating Region
        engineered_df['vsb'] = -engineered_df['vb']  # Source-to-bulk voltage

        # Parameters for Vth calculation from config
        vth_params = settings.get('vth_params')
        engineered_df['vth'] = engineered_df['vsb'].apply(
            lambda x: calculate_vth(x,
                                    vth0=vth_params['vth0'],
                                    gamma=vth_params['gamma'],
                                    phi_f=vth_params['phi_f']))
        print("  - Calculated 'vth' dynamically using body effect formula.")

        # Classify operating region using the calculated Vth
        # Pass relevant columns and Vth calculation params from config
        engineered_df['operating_region'] = engineered_df.apply(
            lambda row: classify_region(row, vth_approx_val=row['vth']), axis=1)  # Use calculated Vth

        print("  - Classified 'operating_region'. Distribution:")
        print(engineered_df['operating_region'].value_counts(normalize=True).round(3))

        return engineered_df

    def load_or_process_data(self, force_reprocess=False):
        """
        Loads processed data if available, otherwise performs full processing pipeline.

        Args:
            force_reprocess (bool): If True, forces reprocessing even if files exist.

        Returns:
            bool: True if data is ready, False otherwise.
        """
        print("\nStarting Data Processing Pipeline...")

        # Check if processed data exists and if we should skip reprocessing
        # Use self.processed_data_dir (which is now a Path object)
        if not force_reprocess and self._check_processed_data_exists(self.processed_data_dir):
            if self.load_processed_data(self.processed_data_dir):
                print("Using existing processed data.")
                return True
            else:
                print("Failed to load existing processed data. Reprocessing.")

        print("No processed data found or reprocessing forced. Starting from raw data.")
        # raw_data_path is now a Path object, DataLoader should be able to handle it
        data_loader = DataLoader(self.raw_data_path)
        raw_df = data_loader.load_raw_data()

        if raw_df is None:
            print("Failed to load raw data. Cannot proceed with preprocessing.")
            return False

        print("Applying filters...")
        self.df = self._apply_filters(raw_df)

        if self.df.empty:
            print("No data remaining after filtering. Cannot proceed.")
            return False

        print("Performing feature engineering...")
        self.filtered_original_df = self._perform_feature_engineering(self.df)

        if self.filtered_original_df.empty:
            print("No data remaining after feature engineering. Cannot proceed.")
            return False

        # Now perform the split and scaling
        print("\nPerforming initial stratified split (CV Pool - Final Test Set)")
        # Get parameters from self.data_config (which was passed in __init__)
        cv_test_split_ratio = settings.get('cv_final_test_split_ratio')
        n_splits_cv = settings.get('num_folds')
        random_state = settings.get('random_state')

        X_full = self.filtered_original_df[self.features_for_model]
        y_full = self.filtered_original_df[self.target_feature].values.reshape(-1, 1)

        X_cv_pool, X_final_test, y_cv_pool, y_final_test, idx_cv_pool, idx_final_test = train_test_split(
            X_full, y_full, self.filtered_original_df.index,
            test_size=cv_test_split_ratio,
            random_state=random_state,
            stratify=self.filtered_original_df[self.stratify_column]
        )

        self.X_cv_original_df = self.filtered_original_df.loc[idx_cv_pool].copy()
        self.X_final_test_original_df = self.filtered_original_df.loc[idx_final_test].copy()

        # Initialize and fit scalers
        self.scaler_X = StandardScaler()
        self.X_cv_scaled = self.scaler_X.fit_transform(X_cv_pool)

        self.scaler_y = StandardScaler()
        self.y_cv_scaled = self.scaler_y.fit_transform(y_cv_pool)

        self.X_final_test_scaled = self.scaler_X.transform(X_final_test)
        self.y_final_test_scaled = self.scaler_y.transform(y_final_test)

        print(f"CV Pool size: {len(self.X_cv_scaled)} samples")
        print(f"Final Test Set size: {len(self.X_final_test_scaled)} samples")

        # Generate CV fold indices
        print(f"\nGenerating {n_splits_cv} stratified folds for CV Pool")
        skf = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state)
        cv_pool_stratify_labels = self.X_cv_original_df[self.stratify_column].values
        self.cv_fold_indices = list(skf.split(self.X_cv_scaled, cv_pool_stratify_labels))
        print(f"Generated {len(self.cv_fold_indices)} CV folds.")

        # save_processed_data now takes a Path object
        self.save_processed_data(self.processed_data_dir)
        print("\nData preparation complete.")
        return True

    # --- Persistence Methods ---
    # Update _check_processed_data_exists and save/load to expect Path objects
    def _check_processed_data_exists(self, path: path):
        """Checks if all expected processed data files exist."""
        expected_files = [
            'x_cv_scaled.pkl', 'y_cv_scaled.pkl', 'x_cv_original_df.pkl', 'cv_fold_indices.pkl',
            'X_final_test_scaled.pkl', 'y_final_test_scaled.pkl', 'X_final_test_original_df.pkl',
            'scaler_x.pkl', 'scaler_y.pkl', 'filtered_original_df.pkl'
        ]
        return all((path / f).exists() for f in expected_files) # Use Path operations

    def save_processed_data(self, save_path: path): # Type hint for clarity
        """Saves processed data and scalers to disk."""
        os.makedirs(save_path, exist_ok=True) # os.makedirs works with Path objects
        # Use Path / operator for joining
        joblib.dump(self.X_cv_scaled, save_path / 'x_cv_scaled.pkl')
        joblib.dump(self.y_cv_scaled, save_path / 'y_cv_scaled.pkl')
        joblib.dump(self.X_cv_original_df, save_path / 'x_cv_original_df.pkl')
        joblib.dump(self.cv_fold_indices, save_path / 'cv_fold_indices.pkl')

        joblib.dump(self.X_final_test_scaled, save_path / 'X_final_test_scaled.pkl')
        joblib.dump(self.y_final_test_scaled, save_path / 'y_final_test_scaled.pkl')
        joblib.dump(self.X_final_test_original_df, save_path / 'X_final_test_original_df.pkl')

        joblib.dump(self.scaler_X, save_path / 'scaler_x.pkl')
        joblib.dump(self.scaler_y, save_path / 'scaler_y.pkl')
        joblib.dump(self.filtered_original_df, save_path / 'filtered_original_df.pkl')
        print(f"Processed data and scalers saved to {save_path}")

    def load_processed_data(self, load_path: path): # Type hint for clarity
        """Loads processed data and scalers from disk."""
        try:
            # Use Path / operator for joining
            self.X_cv_scaled = joblib.load(load_path / 'x_cv_scaled.pkl')
            self.y_cv_scaled = joblib.load(load_path / 'y_cv_scaled.pkl')
            self.X_cv_original_df = joblib.load(load_path / 'x_cv_original_df.pkl')
            self.cv_fold_indices = joblib.load(load_path / 'cv_fold_indices.pkl')

            self.X_final_test_scaled = joblib.load(load_path / 'X_final_test_scaled.pkl')
            self.y_final_test_scaled = joblib.load(load_path / 'y_final_test_scaled.pkl')
            self.X_final_test_original_df = joblib.load(load_path / 'X_final_test_original_df.pkl')

            self.scaler_X = joblib.load(load_path / 'scaler_x.pkl')
            self.scaler_y = joblib.load(load_path / 'scaler_y.pkl')
            self.filtered_original_df = joblib.load(load_path / 'filtered_original_df.pkl')
            print(f"Processed data and scalers loaded from {load_path}")
            return True
        except FileNotFoundError:
            print(f"No processed data found at {load_path}. Cannot load.")
            return False
        except Exception as e:
            print(f"Error loading processed data from {load_path}: {e}")
            return False

    # GETTERS (remain largely the same)
    def get_cv_data(self):
        """Returns scaled CV pool data and fold indices."""
        if self.X_cv_scaled is None or not self.cv_fold_indices:
            raise ValueError("CV data not prepared. Call load_or_process_data() first.")
        return self.X_cv_scaled, self.y_cv_scaled, self.X_cv_original_df, self.cv_fold_indices

    def get_final_test_data(self):
        """Returns scaled final test data and original dataframe for plotting."""
        if self.X_final_test_scaled is None:
            raise ValueError("Final test data not prepared. Call load_or_process_data() first.")
        return self.X_final_test_scaled, self.y_final_test_scaled, self.X_final_test_original_df

    def get_scalers(self):
        """Returns the fitted X and y scalers."""
        if self.scaler_X is None or self.scaler_y is None:
            raise ValueError("Scalers not fitted. Call load_or_process_data() first.")
        return self.scaler_X, self.scaler_y

    def get_features_for_model(self):
        """Returns the list of features used for the model."""
        return self.features_for_model

    def get_filtered_original_data(self):
        """Returns the full original DataFrame after initial filtering and preprocessing."""
        if self.filtered_original_df is None:
            # If not loaded or processed yet, trigger full processing
            print("Full filtered data not available, attempting to process...")
            self.load_or_process_data() # This will use its internal config
        return self.filtered_original_df