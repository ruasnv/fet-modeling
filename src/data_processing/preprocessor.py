# src/data_processing/preprocessor.py
import os
import warnings

import joblib
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.helpers import calculate_vth, classify_region  # Assuming these are robust


class DataPreprocessor:  # Renamed from DataProcessor for clarity of responsibility
    def __init__(self, main_config_path, data_config_path):
        """
        Initializes the DataPreprocessor with paths to configuration files.
        """
        self.main_config = self._load_config(main_config_path)
        self.data_config = self._load_config(data_config_path)

        self.raw_data_path = self.main_config['paths']['raw_data_path']
        self.processed_data_dir = self.main_config['paths']['processed_data_dir']

        self.df = None  # Raw loaded DataFrame
        self.filtered_original_df = None  # Stores the full DataFrame after filtering and feature engineering
        self.scaler_X = None
        self.scaler_y = None

        # Features for the model, now from config
        self.features_for_model = self.data_config['feature_engineering']['input_features']
        if self.data_config['feature_engineering']['include_w_over_l']:
            self.features_for_model.append('wOverL')  # Dynamically add if needed

        self.target_feature = self.data_config['normalization']['target_feature']  # 'log_Id'
        self.stratify_column = self.data_config['region_classification']['stratify_column']  # 'operating_region'

        # Processed data components (will be set after preparation or loading)
        self.X_cv_scaled = None
        self.y_cv_scaled = None
        self.X_cv_original_df = None
        self.cv_fold_indices = []

        self.X_final_test_scaled = None
        self.y_final_test_scaled = None
        self.X_final_test_original_df = None

    def _load_config(self, config_path):
        """Loads a YAML configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _apply_filters(self, df):
        """Applies filters based on data_config."""
        initial_rows = len(df)
        filtered_df = df.copy()

        # Id > 0
        if self.data_config['filtering']['id_greater_than_zero']:
            filtered_df = filtered_df[filtered_df['id'] > 0].copy()
            print(f"  - After Id > 0 filter: {len(filtered_df)} rows")

        # Vd > 0
        if self.data_config['filtering']['vds_greater_than_zero']:
            filtered_df = filtered_df[filtered_df['vd'] > 0].copy()
            print(f"  - After Vd > 0 filter: {len(filtered_df)} rows")

        # Temperature filter
        temp_filter = self.data_config['filtering'].get('temperature_filter')
        if temp_filter is not None:
            filtered_df = filtered_df[filtered_df['temp'] == temp_filter].copy()
            print(f"  - After Temp = {temp_filter}C filter: {len(filtered_df)} rows")

        print(f"Total filtered rows: {len(filtered_df)} (from {initial_rows} initial rows)")
        return filtered_df

    def _perform_feature_engineering(self, df):
        """Performs feature engineering based on data_config."""
        engineered_df = df.copy()

        # Add W/L feature
        if self.data_config['feature_engineering']['include_w_over_l']:
            engineered_df['wOverL'] = engineered_df['w'] / engineered_df['l']
            print("  - Added 'wOverL' feature.")

        # Log transform for Id
        if self.data_config['normalization']['log_transform_id']:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # Ignore log of zero/negative warnings
                engineered_df['log_Id'] = np.log10(engineered_df['id'])
            print("  - Applied log10 transformation to 'id' to create 'log_Id'.")

        # Calculate Vth and Operating Region
        engineered_df['vsb'] = -engineered_df['vb']  # Source-to-bulk voltage

        # Parameters for Vth calculation from config
        vth_params = self.data_config['region_classification']['vth_params']
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
        if not force_reprocess and self._check_processed_data_exists(self.processed_data_dir):
            if self.load_processed_data(self.processed_data_dir):
                print("Using existing processed data.")
                return True
            else:
                print("Failed to load existing processed data. Reprocessing.")

        print("No processed data found or reprocessing forced. Starting from raw data.")
        data_loader = DataLoader(self.raw_data_path)
        raw_df = data_loader.load_raw_data()

        if raw_df is None:
            print("Failed to load raw data. Cannot proceed with preprocessing.")
            return False

        print("Applying filters...")
        self.df = self._apply_filters(raw_df)  # Store the initially filtered data as self.df

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
        cv_test_split_ratio = self.data_config['data_split']['cv_final_test_split_ratio']
        n_splits_cv = self.data_config['data_split']['num_folds']
        random_state = self.data_config['data_split']['random_state']

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
        self.scaler_X = StandardScaler()  # From data_config['normalization']['method'] if you want to generalize
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

        self.save_processed_data(self.processed_data_dir)
        print("\nData preparation complete.")
        return True

    # --- Persistence Methods ---
    def _check_processed_data_exists(self, path):
        """Checks if all expected processed data files exist."""
        expected_files = [
            'x_cv_scaled.pkl', 'y_cv_scaled.pkl', 'x_cv_original_df.pkl', 'cv_fold_indices.pkl',
            'X_final_test_scaled.pkl', 'y_final_test_scaled.pkl', 'X_final_test_original_df.pkl',
            'scaler_x.pkl', 'scaler_y.pkl', 'filtered_original_df.pkl'
        ]
        return all(os.path.exists(os.path.join(path, f)) for f in expected_files)

    def save_processed_data(self, save_path):
        """Saves processed data and scalers to disk."""
        os.makedirs(save_path, exist_ok=True)
        # Note: Using attribute names for consistency
        joblib.dump(self.X_cv_scaled, os.path.join(save_path, 'x_cv_scaled.pkl'))
        joblib.dump(self.y_cv_scaled, os.path.join(save_path, 'y_cv_scaled.pkl'))
        joblib.dump(self.X_cv_original_df, os.path.join(save_path, 'x_cv_original_df.pkl'))
        joblib.dump(self.cv_fold_indices, os.path.join(save_path, 'cv_fold_indices.pkl'))

        joblib.dump(self.X_final_test_scaled, os.path.join(save_path, 'X_final_test_scaled.pkl'))
        joblib.dump(self.y_final_test_scaled, os.path.join(save_path, 'y_final_test_scaled.pkl'))
        joblib.dump(self.X_final_test_original_df, os.path.join(save_path, 'X_final_test_original_df.pkl'))

        joblib.dump(self.scaler_X, os.path.join(save_path, 'scaler_x.pkl'))
        joblib.dump(self.scaler_y, os.path.join(save_path, 'scaler_y.pkl'))
        joblib.dump(self.filtered_original_df, os.path.join(save_path, 'filtered_original_df.pkl'))
        print(f"Processed data and scalers saved to {save_path}")

    def load_processed_data(self, load_path):
        """Loads processed data and scalers from disk."""
        try:
            self.X_cv_scaled = joblib.load(os.path.join(load_path, 'x_cv_scaled.pkl'))
            self.y_cv_scaled = joblib.load(os.path.join(load_path, 'y_cv_scaled.pkl'))
            self.X_cv_original_df = joblib.load(os.path.join(load_path, 'x_cv_original_df.pkl'))
            self.cv_fold_indices = joblib.load(os.path.join(load_path, 'cv_fold_indices.pkl'))

            self.X_final_test_scaled = joblib.load(os.path.join(load_path, 'X_final_test_scaled.pkl'))
            self.y_final_test_scaled = joblib.load(os.path.join(load_path, 'y_final_test_scaled.pkl'))
            self.X_final_test_original_df = joblib.load(os.path.join(load_path, 'X_final_test_original_df.pkl'))

            self.scaler_X = joblib.load(os.path.join(load_path, 'scaler_x.pkl'))
            self.scaler_y = joblib.load(os.path.join(load_path, 'scaler_y.pkl'))
            self.filtered_original_df = joblib.load(os.path.join(load_path, 'filtered_original_df.pkl'))
            print(f"Processed data and scalers loaded from {load_path}")
            return True
        except FileNotFoundError:
            print(f"No processed data found at {load_path}. Cannot load.")
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
            self.load_or_process_data()
        return self.filtered_original_df