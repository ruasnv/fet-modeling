# src/dataProcessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
import joblib
import os
import warnings

from src.helpers import calculate_vth, classify_region


class DataProcessor:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.df = None  # Raw loaded DataFrame
        self.filtered_original_df = None  # Stores the full DataFrame after filtering and feature engineering
        self.scaler_X = None
        self.scaler_y = None
        #The feature vector for the model.
        #TODO: ADD "WoverL"
        self.features_for_model = ['vg', 'vd', 'vb', 'w', 'l']
        self.target_feature = 'log_Id'
        self.stratify_column = 'operating_region'

        self.X_cv_scaled = None
        self.y_cv_scaled = None
        self.X_cv_original_df = None
        self.cv_fold_indices = []

        self.X_final_test_scaled = None
        self.y_final_test_scaled = None
        self.X_final_test_original_df = None

    def _load_raw_data(self):
        """Loads raw data."""
        print(f"Loading raw data from {self.raw_data_path}...")
        try:
            self.df = pd.read_csv(self.raw_data_path)
            #Vs = 0. Source terminal is grounded.
            self.df['vgs'] = self.df['vg']
            self.df['vds'] = self.df['vd']
        except FileNotFoundError:
            print(
                f"Error: Raw data file not found at {self.raw_data_path}. Please ensure it's in the 'data' directory.")
            self.df = None  # Ensure df is None if file not found
            return

    def _filter_data(self, temp_filter=27.0):
        """Filters the data"""
        if self.df is None:
            self._load_raw_data()
            if self.df is None:  # If raw data still not loaded
                return

        initial_rows = len(self.df)
        self.filtered_original_df = self.df[
            (self.df['id'] > 0) &
            (self.df['vd'] > 0) &
            (self.df['temp'] == temp_filter)
            ].copy()

        print(
            f"Filtered data: {len(self.filtered_original_df)} rows remaining (from {initial_rows} initial rows) after Id>0, Vd>0, and Temp={temp_filter}C filtering.")

    def _engineer_features(self):
        """Engineers new features."""
        if self.filtered_original_df is None:
            self._filter_data()
            if self.filtered_original_df is None:  # Filtered data still not available
                return

        self.filtered_original_df['wOverL'] = self.filtered_original_df['w'] / self.filtered_original_df[
            'l']

        #Log transformation for Id. Wrapped in warning catch block. log10 is used.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.filtered_original_df['log_Id'] = np.log10(self.filtered_original_df['id'])

        #Calculates the Vth dynamically based on the body effect, function calculate_vth is in helpers.py
        self.filtered_original_df['vsb'] = -self.filtered_original_df['vb']
        self.filtered_original_df['vth'] = calculate_vth(self.filtered_original_df['vsb'])

        #Find operating regions with new Vth using the classify_region helper function located in helpers.py.
        self.filtered_original_df['operating_region'] = self.filtered_original_df.apply(classify_region, axis=1)

        print("Operating region distribution:")
        print(self.filtered_original_df['operating_region'].value_counts(normalize=True).round(3))

    def prepare_cv_and_final_test_sets(self, cv_test_split_ratio=0.1, n_splits_cv=10, random_state=42,save_path="data/processed_data"):
        """
        Prepares the data for cross-validation and a final held-out test set.
        Prints messages about the data preparation process.
        """
        print("\nStarting Data Preparation for 10-fold Cross Validation and Separate Final Test Set")
        self._engineer_features()

        if self.filtered_original_df is None or self.filtered_original_df.empty:
            print(
                "Error: No data available after filtering and feature engineering. Cannot prepare CV and final test sets.")
            return


        X_full = self.filtered_original_df[self.features_for_model]
        y_full = self.filtered_original_df[self.target_feature].values.reshape(-1, 1)

        print(f"\nPerforming initial stratified split (CV Pool - Final Test Set with ratio={cv_test_split_ratio})")
        X_cv_pool, X_final_test, y_cv_pool, y_final_test, idx_cv_pool, idx_final_test = train_test_split(
            X_full, y_full, self.filtered_original_df.index,  # Use filtered_original_df.index for splitting
            test_size=cv_test_split_ratio,
            random_state=random_state,
            stratify=self.filtered_original_df[self.stratify_column]  # Use filtered_original_df for stratification
            # Stratified split ensured protection of the data distribution over splits. Protection over the stratified_column field.
            # stratify_column is Operating Regions in this case.
        )

        self.X_cv_original_df = self.filtered_original_df.loc[idx_cv_pool].copy()
        self.X_final_test_original_df = self.filtered_original_df.loc[idx_final_test].copy()

        #Scaling X and y
        #TODO: Don't forget inverse transform when plotting.
        # np.power(10, inverse_transform(predicted_log_id_scaled))

        self.scaler_X = StandardScaler()
        self.X_cv_scaled = self.scaler_X.fit_transform(X_cv_pool)

        self.scaler_y = StandardScaler()
        self.y_cv_scaled = self.scaler_y.fit_transform(y_cv_pool)

        self.X_final_test_scaled = self.scaler_X.transform(X_final_test)
        self.y_final_test_scaled = self.scaler_y.transform(y_final_test)

        print(f"CV Pool size: {len(self.X_cv_scaled)} samples")
        print(f"Final Test Set size: {len(self.X_final_test_scaled)} samples")

        #For Debugging if needed
        #print("Sample:")
        #print(self.filtered_original_df[['vg', 'vb', 'vsb', 'vth', 'vgs', 'vds', 'operating_region']].sample(5))

        #Spliting the CV dataset into 10 folds for cross validation
        print(f"\nGenerating {n_splits_cv} stratified folds for CV Pool")
        skf = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state)
        cv_pool_stratify_labels = self.X_cv_original_df[self.stratify_column].values
        self.cv_fold_indices = list(skf.split(self.X_cv_scaled, cv_pool_stratify_labels))
        print(f"Generated {len(self.cv_fold_indices)} CV folds.")
        self.save_processed_data(save_path)
        print("\nData preparation complete")

    #GETTERS
    def get_cv_data(self):
        """Returns scaled CV pool data and fold indices."""
        if self.X_cv_scaled is None or not self.cv_fold_indices:
            raise ValueError("CV data not prepared. Call prepare_cv_and_final_test_sets() first.")
        return self.X_cv_scaled, self.y_cv_scaled, self.X_cv_original_df, self.cv_fold_indices

    def get_final_test_data(self):
        """Returns scaled final test data and original dataframe for plotting."""
        if self.X_final_test_scaled is None:
            raise ValueError("Final test data not prepared. Call prepare_cv_and_final_test_sets() first.")
        return self.X_final_test_scaled, self.y_final_test_scaled, self.X_final_test_original_df

    def get_scalers(self):
        """Returns the fitted X and y scalers."""
        if self.scaler_X is None or self.scaler_y is None:
            raise ValueError("Scalers not fitted. Call prepare_cv_and_final_test_sets() first.")
        return self.scaler_X, self.scaler_y

    def get_features_for_model(self):
        """Returns the list of features used for the model."""
        return self.features_for_model

    def get_filtered_original_data(self):  # New public method to access the full filtered data
        """Returns the full original DataFrame after initial filtering and preprocessing."""
        if self.filtered_original_df is None:
            # If not loaded or processed yet, trigger processing
            self._engineer_features()  # This will call _filter_data and _load_raw_data if needed
        return self.filtered_original_df

    def save_processed_data(self, save_path):
        """Saves processed data and scalers to disk."""
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(self.X_cv_scaled, os.path.join(save_path, 'x_cv_scaled.pkl'))
        joblib.dump(self.y_cv_scaled, os.path.join(save_path, 'y_cv_scaled.pkl'))
        joblib.dump(self.X_cv_original_df, os.path.join(save_path, 'x_cv_original_df.pkl'))
        joblib.dump(self.cv_fold_indices, os.path.join(save_path, 'cv_fold_indices.pkl'))

        joblib.dump(self.X_final_test_scaled, os.path.join(save_path, 'X_final_test_scaled.pkl'))
        joblib.dump(self.y_final_test_scaled, os.path.join(save_path, 'y_final_test_scaled.pkl'))
        joblib.dump(self.X_final_test_original_df, os.path.join(save_path, 'X_final_test_original_df.pkl'))

        joblib.dump(self.scaler_X, os.path.join(save_path, 'scaler_x.pkl'))
        joblib.dump(self.scaler_y, os.path.join(save_path, 'scaler_y.pkl'))
        joblib.dump(self.filtered_original_df,
                    os.path.join(save_path, 'filtered_original_df.pkl'))  # Save the full filtered data
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
            self.filtered_original_df = joblib.load(
                os.path.join(load_path, 'filtered_original_df.pkl'))  # Load the full filtered data
            print(f"Processed data and scalers loaded from {load_path}")
            return True
        except FileNotFoundError:
            print(f"No processed data found at {load_path}. Will re-process.")
            return False

