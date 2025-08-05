# src/data_processing/preprocessor.py
import os
import warnings
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from .data_loader import DataLoader
from src.utils.helpers import calculate_vth, classify_region
from src.config import settings


class DataPreprocessor:
    def __init__(self):
        """
        Initializes the DataPreprocessor with the complete merged configuration data.
        """
        self.raw_data_path = settings.get('paths.raw_data_path')
        self.processed_data_dir = settings.get('paths.processed_data_dir')

        self.df = None
        self.filtered_original_df = None
        self.scaler_X = None
        self.scaler_y = None

        # Get the correct input features from the config
        self.features_for_model = settings.get('feature_engineering.input_features')
        self.target_feature = settings.get('feature_engineering.target_feature')
        self.stratify_column = settings.get('region_classification.stratify_column')

        self.X_cv_scaled = None
        self.y_cv_scaled = None
        self.X_cv_original_df = None
        self.cv_fold_indices = []

        self.X_final_test_scaled = None
        self.y_final_test_scaled = None
        self.X_final_test_original_df = None

    def _apply_filters(self, df):
        """Applies filters based on data_config."""
        initial_rows = len(df)
        filtered_df = df.copy()

        if settings.get('data_filter.id_greater_than_zero'):
            filtered_df = filtered_df[filtered_df['id'] > 0].copy()
            print(f"  - After Id > 0 filter: {len(filtered_df)} rows")

        # The Vds filter is applied after Vds is calculated in feature engineering.
        temp_filter = settings.get('data_filter.temperature_filter')
        if temp_filter is not None:
            filtered_df = filtered_df[np.isclose(filtered_df['temp'], temp_filter)].copy()
            print(f"  - After Temp = {temp_filter}°C filter: {len(filtered_df)} rows")

        print(f"Total filtered rows: {len(filtered_df)} (from {initial_rows} initial rows)")
        return filtered_df

    def _perform_feature_engineering(self, df):
        """Performs feature engineering based on data_config."""
        engineered_df = df.copy()

        # Calculate derived quantities for internal use (like operating region classification)
        engineered_df['vgs'] = engineered_df['vg'] - engineered_df['vs']
        engineered_df['vds'] = engineered_df['vd'] - engineered_df['vs']
        engineered_df['vbs'] = engineered_df['vb'] - engineered_df['vs']
        print("  - Calculated Vgs, Vds, and Vbs features.")

        # Now apply the Vds filter using the newly calculated column
        if settings.get('data_filter.vds_greater_than_zero'):
            engineered_df = engineered_df[engineered_df['vds'] > 0].copy()
            print(f"  - After Vds > 0 filter: {len(engineered_df)} rows")

        # Create 'wOverL' feature
        engineered_df['wOverL'] = engineered_df['w'] / engineered_df['l']
        print("  - Added 'wOverL' feature.")

        if settings.get('normalization.log_transform_id'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                engineered_df['log_Id'] = np.log10(engineered_df['id'])
            print("  - Applied log10 transformation to 'id' to create 'log_Id'.")

            # --- FIX: Explicitly remove any rows with non-finite values after log transformation
            initial_rows_after_log = len(engineered_df)
            # Replace inf/-inf with NaN first, then drop NaN
            engineered_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            engineered_df.dropna(subset=['log_Id'], inplace=True)
            rows_removed_due_to_log = initial_rows_after_log - len(engineered_df)
            if rows_removed_due_to_log > 0:
                print(f"  - Cleaned non-finite values from 'log_Id'. Removed {rows_removed_due_to_log} rows.")
            else:
                print(f"  - No non-finite values found in 'log_Id' after transformation.")


        engineered_df['vsb'] = engineered_df['vs'] - engineered_df['vb']
        engineered_df['vth'] = engineered_df['vsb'].apply(
            lambda x: calculate_vth(x,
                                    vth0=settings.get('vth_params.vth0'),
                                    gamma=settings.get('vth_params.gamma'),
                                    phi_f=settings.get('vth_params.phi_f')))
        print("  - Calculated 'vth' dynamically using body effect formula.")

        engineered_df['operating_region'] = engineered_df.apply(
            lambda row: classify_region(row, vth_approx_val=row['vth']), axis=1)
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
        processed_data_path = Path(self.processed_data_dir)

        if not force_reprocess and self._check_processed_data_exists(processed_data_path):
            if self.load_processed_data(processed_data_path):
                print(f" Using existing processed data from: {processed_data_path}")
                return True
            else:
                print(" Failed to load existing processed data. Reprocessing.")

        print("No processed data found or reprocessing forced. Starting from raw data.")
        data_loader = DataLoader(self.raw_data_path)
        raw_df = data_loader.load_raw_data()

        if raw_df is None:
            print(" Failed to load raw data. Cannot proceed with preprocessing.")
            return False

        print("Applying filters...")
        self.df = self._apply_filters(raw_df)

        if self.df.empty:
            print(" No data remaining after filtering. Cannot proceed.")
            return False

        print("Performing feature engineering...")
        self.filtered_original_df = self._perform_feature_engineering(self.df)

        if self.filtered_original_df.empty:
            print(" No data remaining after feature engineering. Cannot proceed.")
            return False

        print("\nPerforming initial stratified split (CV Pool - Final Test Set)")
        cv_test_split_ratio = settings.get('data_split.cv_final_test_split_ratio')
        n_splits_cv = settings.get('data_split.num_folds')
        random_state = settings.get('global_settings.random_state')

        # Use the correct features_for_model from the config
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

        self.scaler_X = StandardScaler()
        self.X_cv_scaled = self.scaler_X.fit_transform(X_cv_pool)

        self.scaler_y = StandardScaler()
        self.y_cv_scaled = self.scaler_y.fit_transform(y_cv_pool)

        self.X_final_test_scaled = self.scaler_X.transform(X_final_test)
        self.y_final_test_scaled = self.scaler_y.transform(y_final_test)

        print(f"CV Pool size: {len(self.X_cv_scaled)} samples")
        print(f"Final Test Set size: {len(self.X_final_test_scaled)} samples")

        print(f"\nGenerating {n_splits_cv} stratified folds for CV Pool")
        skf = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state)
        cv_pool_stratify_labels = self.X_cv_original_df[self.stratify_column].values
        self.cv_fold_indices = list(skf.split(self.X_cv_scaled, cv_pool_stratify_labels))
        print(f"Generated {len(self.cv_fold_indices)} CV folds.")

        self.save_processed_data(processed_data_path)
        print("\nData preparation complete.")
        return True

    def _check_processed_data_exists(self, path: Path):
        """Checks if all expected processed data files exist."""
        expected_files = [
            'x_cv_scaled.pkl', 'y_cv_scaled.pkl', 'x_cv_original_df.pkl', 'cv_fold_indices.pkl',
            'X_final_test_scaled.pkl', 'y_final_test_scaled.pkl', 'X_final_test_original_df.pkl',
            'scaler_x.pkl', 'scaler_y.pkl', 'filtered_original_df.pkl'
        ]
        return all((path / f).exists() for f in expected_files)

    def save_processed_data(self, save_path: Path):
        """Saves processed data and scalers to disk."""
        os.makedirs(save_path, exist_ok=True)
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

    def load_processed_data(self, load_path: Path):
        """Loads processed data and scalers from disk."""
        try:
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
            print("Full filtered data not available, attempting to process...")
            self.load_or_process_data()
        return self.filtered_original_df
