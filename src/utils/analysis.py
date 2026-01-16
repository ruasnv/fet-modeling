# src/utils/analysis.py
import numpy as np
import pandas as pd
import torch

def suggest_best_worst_cases(df_region, region_name, predictions_for_region, min_points_for_case=5):
    """
    Suggests a 'best' and 'worst' case scenario based on prediction error.
    """
    if df_region.empty:
        print(f"No data found for region: {region_name}. Cannot suggest best/worst cases.")
        return None, None

    df_region = df_region.copy() # Avoid SettingWithCopy warning
    df_region['prediction_error'] = (df_region['id'].values - predictions_for_region.flatten()) ** 2

    # Group by unique case parameters to find the number of data points per case
    group_cols = ['w', 'l', 'vg', 'vb']
    case_counts = df_region.groupby(group_cols).size().reset_index(name='count')

    # Filter out cases with fewer than the minimum required data points
    valid_cases = case_counts[case_counts['count'] >= min_points_for_case]

    if valid_cases.empty:
        print(f"No cases in '{region_name}' region have at least {min_points_for_case} data points. Skipping.")
        return None, None

    # find the best and worst cases only from the valid cases
    grouped_errors = df_region.groupby(group_cols)['prediction_error'].mean().reset_index()

    # Merge the valid cases with their corresponding mean error
    valid_cases_with_errors = pd.merge(valid_cases, grouped_errors, on=group_cols)

    if valid_cases_with_errors.empty:
        return None, None

    # Find the combinations with the minimum and maximum MAE
    best_case_series = valid_cases_with_errors.loc[valid_cases_with_errors['prediction_error'].idxmin()]
    worst_case_series = valid_cases_with_errors.loc[valid_cases_with_errors['prediction_error'].idxmax()]

    # Calculate the Vds range for the entire region
    region_min_vds = df_region['vd'].min()
    region_max_vds = df_region['vd'].max()
    dynamic_vds_range = [region_min_vds, region_max_vds]

    def format_case_for_plotter(case_series, label_prefix):
        return {
            'device_size': [case_series['w'] * 1e6, case_series['l'] * 1e6],
            'vbs_val': case_series['vb'],
            'plot_type': "Id_Vds_fixed_Vgs",
            'fixed_vgs_vals': [case_series['vg']],
            'Vds_range': dynamic_vds_range,
            'label': f"{label_prefix} - W={case_series['w'] * 1e6:.1f}µm, L={case_series['l'] * 1e6:.1f}µm, Vbs={case_series['vb']:.1f}V, Vgs={case_series['vg']:.2f}V"
        }

    best_case_dict = format_case_for_plotter(best_case_series, "Best Case")
    worst_case_dict = format_case_for_plotter(worst_case_series, "Worst Case")

    return best_case_dict, worst_case_dict

def determine_characteristic_plot_cases(model, full_filtered_original_df, scaler_X, scaler_y, features_for_model, device):
    """
    Determines best and worst case scenarios across operating regions.
    """
    model.eval()
    full_filtered_original_df = full_filtered_original_df.reset_index(drop=True)

    X_full = full_filtered_original_df[features_for_model]
    X_full_scaled = scaler_X.transform(X_full)
    X_tensor = torch.tensor(X_full_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()
        full_predictions = scaler_y.inverse_transform(y_pred_scaled)
        full_predictions = np.power(10, full_predictions).flatten()

    regions = ["Cut-off", "Linear", "Saturation"]
    plot_cases_by_region = {}

    for region in regions:
        df_region_subset = full_filtered_original_df[full_filtered_original_df['operating_region'] == region].copy()

        if df_region_subset.empty:
            continue

        original_indices = df_region_subset.index.values
        predictions_for_region = full_predictions[original_indices]

        best_case, worst_case = suggest_best_worst_cases(
            df_region_subset, region, predictions_for_region
        )

        if best_case and worst_case:
            plot_cases_by_region[region] = [best_case, worst_case]
            
    return plot_cases_by_region