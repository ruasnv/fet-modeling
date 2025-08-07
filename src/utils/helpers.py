# src/utils/helpers.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from src.config import settings
import torch

def setup_environment():
    """
    Sets up matplotlib environment based on global settings from main_config.yaml
    and creates the base report output directory.

    Args:
    """
    report_output_dir = settings.get('paths.report_output_dir')
    # Apply Matplotlib style and parameters
    plt.style.use(settings.get('global_settings.matplotlib_style'))
    plt.rcParams['figure.figsize'] = settings.get('global_settings.figure_figsize')
    plt.rcParams['font.size'] = settings.get('global_settings.font_size')
    plt.rcParams['axes.labelsize'] =settings.get('global_settings.axes_labelsize')
    plt.rcParams['axes.titlesize'] = settings.get('global_settings.axes_titlesize')
    plt.rcParams['legend.fontsize'] = settings.get('global_settings.legend_fontsize')

    # Ensure the base report output directory exists
    os.makedirs(report_output_dir, exist_ok=True)
    print(f"Report output base directory created/ensured: {report_output_dir}")

def suggest_best_worst_cases(df_region, region_name, predictions_for_region, min_points_for_case=5):
    """
    Suggests a 'best' and 'worst' case scenario for a given operating region
    based on the average prediction error, ensuring each case has a minimum number of data points.

    Args:
        df_region (pd.DataFrame): The DataFrame subset for the specific region.
        region_name (str): The name of the operating region to analyze.
        predictions_for_region (np.ndarray): The model's predictions for the data in df_region.
        min_points_for_case (int): The minimum number of data points a case must have to be considered.

    Returns:
        tuple: (best_case_dict, worst_case_dict)
               Each dict contains device parameters for plotting.
               Returns (None, None) if no valid cases are found.
    """
    if df_region.empty:
        print(f"No data found for region: {region_name}. Cannot suggest best/worst cases.")
        return None, None

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
        # The Vds range is now set to the dynamic range calculated for the entire region
        # This will ensure the predicted line covers the full sweep of the available data.
        specific_case_subset = df_region[
            (np.isclose(df_region['w'], case_series['w'], atol=1e-9)) &
            (np.isclose(df_region['l'], case_series['l'], atol=1e-9)) &
            (np.isclose(df_region['vg'], case_series['vg'], atol=1e-2)) &
            (np.isclose(df_region['vb'], case_series['vb'], atol=1e-2))
            ]

        return {
            'device_size': [case_series['w'] * 1e6, case_series['l'] * 1e6],
            'vbs_val': case_series['vb'],
            'plot_type': "Id_Vds_fixed_Vgs",
            'fixed_vgs_vals': [case_series['vg']],
            'Vds_range': dynamic_vds_range,  # Use the dynamic range here
            'label': f"{label_prefix} - W={case_series['w'] * 1e6:.1f}µm, L={case_series['l'] * 1e6:.1f}µm, Vbs={case_series['vb']:.1f}V, Vgs={case_series['vg']:.2f}V"
        }

    best_case_dict = format_case_for_plotter(best_case_series, "Best Case")
    worst_case_dict = format_case_for_plotter(worst_case_series, "Worst Case")

    return best_case_dict, worst_case_dict

def determine_characteristic_plot_cases(model, full_filtered_original_df, scaler_X, scaler_y, features_for_model,
                                        device):
    """
    Determines best and worst case scenarios for Id-Vds characteristic plots
    across different operating regions based on model prediction error.
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
        # Get the DataFrame for the current region
        df_region_subset = full_filtered_original_df[full_filtered_original_df['operating_region'] == region].copy()

        if df_region_subset.empty:
            print(f"Skipping characteristic plot cases for {region} due to insufficient data.")
            continue

        # Get the indices of the subset, which are now correctly 0-based
        original_indices = df_region_subset.index.values

        # Use those original indices to slice the full_predictions array
        predictions_for_region = full_predictions[original_indices]

        best_case, worst_case = suggest_best_worst_cases(
            df_region_subset,
            region,
            predictions_for_region
        )

        if best_case and worst_case:
            plot_cases_by_region[region] = [best_case, worst_case]
        else:
            print(f"Skipping characteristic plot cases for {region} due to insufficient data.")

    return plot_cases_by_region


#TODO: This method of calculating the Vth for region separation was added later.
# It used the Body effect formula with default parameters for the NMOS-HV devices.
# However the region distribution changes significantly when this method is used.
# I suggest to check this theoretically with a domain expert.
def calculate_vth(vsb, vth0=0.7, gamma=0.4, phi_f=0.4):
    """
    Calculates the threshold voltage (Vth) using the body effect formula.

    Args:
        vsb (pd.Series or float): Source-to-bulk voltage (V_SB).
        vth0 (float): Zero-bias threshold voltage (Vth when Vsb=0).
        gamma (float): Body effect coefficient.
        phi_f (float): Fermi potential. In the formula, 2*phi_f typically refers to 2 * kT/q * ln(Na/ni).
                       Ensure this 'phi_f' parameter aligns with the expected '2*phi_f' in the formula or is phi_f itself.

    Returns:
        pd.Series or float: Calculated threshold voltage.
    """
    # Common formula for body effect: Vth = Vth0 + gamma * (sqrt(2*phi_f + V_SB) - sqrt(2*phi_f))
    # `2*phi_f + vsb` could become negative. Clipping to a small positive value prevents NaNs.
    # The value 1e-9 is an arbitrary small positive number to avoid log(0) and sqrt(0) issues.
    sqrt_term_arg = 2 * phi_f + vsb
    sqrt_term_arg = np.maximum(sqrt_term_arg, 1e-9)
    return vth0 + gamma * (np.sqrt(sqrt_term_arg) - np.sqrt(2 * phi_f))


def classify_region(row, vth_approx_val):
    """
    Classifies the operating region of a MOSFET based on Vgs, Vds, and Vth.

    Args:
        row (pandas.Series): A row from the DataFrame containing 'vgs' and 'vds'.
        vth_approx_val (float): The calculated or approximate threshold voltage (Vth) for the device
                                under the given body bias.

    Returns:
        str: 'Cut-off', 'Linear', or 'Saturation'.
    """

    # Assumption can be made since Source voltage is 0
    vgs = row['vg']
    vds = row['vd']

    # Cutoff Region: Vgs <= Vth
    if vgs <= vth_approx_val:
        return 'Cut-off'
    # Linear Region: Vds < (Vgs - Vth)
    elif vds < (vgs - vth_approx_val):
        return 'Linear'
    # Saturation Region: Vds >= (Vgs - Vth)
    else:
        return 'Saturation'