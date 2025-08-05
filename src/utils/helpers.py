# src/utils/helpers.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from src.config import settings


def setup_environment():
    """
    Sets up matplotlib environment based on global settings from main_config.yaml
    and creates the base report output directory.

    Args:
    """
    # Load global settings from main_config.yaml
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
    print(f"Matplotlib backend: {plt.get_backend()}")


def suggest_best_worst_cases(df, region_name):
    """
    Suggests a 'best' and 'worst' case scenario for a given operating region
    based on average Id across unique (W, L, Vg) combinations.

    'Best' is defined as the combination with the highest average Id.
    'Worst' is defined as the combination with the lowest average Id.

    Args:
        df (pd.DataFrame): The filtered and feature-engineered DataFrame.
        region_name (str): The specific operating region to analyze (e.g., "Cut-off", "Linear", "Saturation").

    Returns:
        tuple: (best_case_dict, worst_case_dict)
               Each dict contains 'W', 'L', 'Vg_const', 'Vbs_val', 'Vds_range', 'label', 'plot_type'.
               Returns (None, None) if no data is found.
    """
    df_region = df[df['operating_region'] == region_name].copy()

    if df_region.empty:
        print(f"No data found for region: {region_name}. Cannot suggest best/worst cases.")
        return None, None

    # Group by W, L, Vg, Vb triplets and calculate mean Id
    # Including 'vb' in grouping to ensure specific Vbs values are considered.
    group_cols = ['w', 'l', 'vg', 'vb']
    grouped = df_region.groupby(group_cols)['id'].mean().reset_index()

    if grouped.empty:
        print(f"No grouped data found for region: {region_name}. Cannot suggest best/worst cases.")
        return None, None

    # Find the max/min average Id combinations (in Amperes)
    best_case_series = grouped.loc[grouped['id'].idxmax()]
    worst_case_series = grouped.loc[grouped['id'].idxmin()]

    def format_case_for_plotter(case_series, label_prefix):
        # Extract the specific row's Vds range for this combination
        specific_case_subset = df_region[
            (np.isclose(df_region['w'], case_series['w'], atol=1e-9)) &
            (np.isclose(df_region['l'], case_series['l'], atol=1e-9)) &
            (np.isclose(df_region['vg'], case_series['vg'], atol=1e-2)) &
            (np.isclose(df_region['vb'], case_series['vb'], atol=1e-2))
        ]

        # Use actual Vds range from the data for the selected combination
        # Fallback to a default range if subset is empty or Vds range is invalid
        case_min_vds = specific_case_subset['vd'].min() if not specific_case_subset.empty and not specific_case_subset['vd'].empty else 0.0
        case_max_vds = specific_case_subset['vd'].max() if not specific_case_subset.empty and not specific_case_subset['vd'].empty else 3.3

        # Ensure min_vds is less than or equal to max_vds
        if case_min_vds > case_max_vds:
            case_min_vds, case_max_vds = 0.0, 3.3 # Fallback to default if range is inverted

        return {
            'device_size': [case_series['w'] * 1e6, case_series['l'] * 1e6], # Convert to um
            'vbs_val': case_series['vb'],
            'plot_type': "Id_Vds_fixed_Vgs", # Always Id-Vds for these cases
            'fixed_vgs_vals': [case_series['vg']], # The Vg for this specific case
            'Vds_range': [case_min_vds, case_max_vds],
            'label': f"{label_prefix} - W={case_series['w'] * 1e6:.1f}µm, L={case_series['l'] * 1e6:.1f}µm, Vbs={case_series['vb']:.1f}V, Vgs={case_series['vg']:.2f}V"
        }

    best_case_dict = format_case_for_plotter(best_case_series, "Best Case")
    worst_case_dict = format_case_for_plotter(worst_case_series, "Worst Case")

    return best_case_dict, worst_case_dict


def determine_characteristic_plot_cases(full_filtered_original_df):
    """
    Determines best and worst case scenarios for Id-Vds characteristic plots
    across different operating regions.

    Args:
        full_filtered_original_df (pd.DataFrame): The preprocessed DataFrame
            containing 'operating_region', 'w', 'l', 'vg', 'id', 'vd', 'vb'.

    Returns:
        dict: A dictionary structured for plotting, where keys are operating regions
              and values are lists of 'best' and 'worst' case dictionaries.
              Each inner dict is formatted for direct use by Plotter.
    """
    regions = ["Cut-off", "Linear", "Saturation"]
    plot_cases_by_region = {}

    for region in regions:
        best_case, worst_case = suggest_best_worst_cases(full_filtered_original_df, region)
        if best_case and worst_case:
            plot_cases_by_region[region] = [best_case, worst_case]
        else:
            print(f"Skipping characteristic plot cases for {region} due to insufficient data.")

    return plot_cases_by_region



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
    # Here, `vsb` directly represents V_SB (Source-Bulk voltage).

    # Ensure the argument to sqrt is non-negative. If `vsb` can be negative (e.g., if vb is positive),
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
    #Assumption can be made since Source voltage is 0
    vgs = row['vg']
    vds = row['vd']

    # Cut-off Region: Vgs <= Vth
    if vgs <= vth_approx_val:
        return 'Cut-off'
    # Linear (Triode) Region: Vds < (Vgs - Vth)
    elif vds < (vgs - vth_approx_val):
        return 'Linear'
    # Saturation Region: Vds >= (Vgs - Vth)
    else:
        return 'Saturation'