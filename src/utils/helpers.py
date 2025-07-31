# src/utils/helpers.py

import pandas as pd
import numpy as np
import os
import yaml
import warnings
import matplotlib.pyplot as plt


def load_config(config_path):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        return {}


def setup_environment(main_config_path):
    """
    Sets up matplotlib environment based on global settings from main_config.yaml
    and creates the base report output directory.

    Args:
        main_config_path (str): Path to the main_config.yaml file.
    """
    # Load global settings from main_config.yaml
    main_config = load_config(main_config_path)
    global_settings = main_config.get('global_settings', {})
    report_output_dir = main_config['paths']['report_output_dir']

    # Apply Matplotlib style and parameters
    plt.style.use(global_settings.get('matplotlib_style', 'default'))
    plt.rcParams['figure.figsize'] = global_settings.get('figure_figsize', [10, 8])
    plt.rcParams['font.size'] = global_settings.get('font_size', 8)
    plt.rcParams['axes.labelsize'] = global_settings.get('axes_labelsize', 10)  # Added for better control
    plt.rcParams['axes.titlesize'] = global_settings.get('axes_titlesize', 12)  # Added for better control
    plt.rcParams['legend.fontsize'] = global_settings.get('legend_fontsize', 8)  # Added for better control

    # Configure warning filtering
    if global_settings.get('ignore_warnings', False):
        warnings.filterwarnings('ignore')  # Use with caution!

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
               Each dict contains 'W', 'L', 'Vg_const', 'Vds_range', 'label'.
    """
    df_region = df[df['operating_region'] == region_name].copy()  # Work on a copy to avoid SettingWithCopyWarning

    if df_region.empty:
        print(f"No data found for region: {region_name}. Cannot suggest best/worst cases.")
        return None, None

    # Group by W, L, Vg triplets and calculate mean Id
    group_cols = ['w', 'l', 'vg']
    grouped = df_region.groupby(group_cols)['id'].mean().reset_index()

    if grouped.empty:
        print(f"No grouped data found for region: {region_name}. Cannot suggest best/worst cases.")
        return None, None

    # Find the max/min average Id combinations (in Amperes)
    best_case = grouped.loc[grouped['id'].idxmax()]
    worst_case = grouped.loc[grouped['id'].idxmin()]

    # Determine the actual Vds range observed for this specific W,L,Vg combination
    # This ensures the plot range matches the available data better.
    # If the combination doesn't exist, fall back to a default wide range.
    min_vds = df_region['vd'].min()  # Consider overall min/max Vd in the region for the synthetic range
    max_vds = df_region['vd'].max()

    def format_case(case_series):
        # Extract the specific row's Vds range
        specific_case_subset = df_region[
            (np.isclose(df_region['w'], case_series['w'], atol=1e-9)) &
            (np.isclose(df_region['l'], case_series['l'], atol=1e-9)) &
            (np.isclose(df_region['vg'], case_series['vg'], atol=1e-2))
            ]

        # Use actual Vds range from the data for the selected W, L, Vg combination
        case_min_vds = specific_case_subset['vd'].min() if not specific_case_subset.empty else min_vds
        case_max_vds = specific_case_subset['vd'].max() if not specific_case_subset.empty else max_vds

        return {
            'W': case_series['w'],
            'L': case_series['l'],
            'Vg_const': case_series['vg'],
            'Vds_range': [case_min_vds, case_max_vds],  # Dynamically set Vds range
            'label': f"W={case_series['w'] * 1e6:.1f}µm, L={case_series['l'] * 1e6:.1f}µm, Vg={case_series['vg']:.2f}V"
        }

    return format_case(best_case), format_case(worst_case)


def determine_characteristic_plot_cases(full_filtered_original_df):
    """
    Determines best and worst case scenarios for Id-Vds characteristic plots
    across different operating regions.

    Args:
        full_filtered_original_df (pd.DataFrame): The preprocessed DataFrame
            containing 'operating_region', 'w', 'l', 'vg', 'id', 'vd'.

    Returns:
        dict: A dictionary structured for plotting, where keys are operating regions
              and values are lists of 'best' and 'worst' case dictionaries.
    """
    # Define the regions we expect to analyze
    regions = ["Cut-off", "Linear", "Saturation"]
    plot_cases = {}

    for region in regions:
        best_case, worst_case = suggest_best_worst_cases(full_filtered_original_df, region)
        if best_case and worst_case:  # Ensure cases were successfully found
            plot_cases[region] = [best_case, worst_case]
        else:
            print(f"Skipping characteristic plot cases for {region} due to insufficient data.")

    return plot_cases


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
    vgs = row['vgs']
    vds = row['vds']

    # Cut-off Region: Vgs <= Vth
    if vgs <= vth_approx_val:
        return 'Cut-off'
    # Linear (Triode) Region: Vds < (Vgs - Vth)
    elif vds < (vgs - vth_approx_val):
        return 'Linear'
    # Saturation Region: Vds >= (Vgs - Vth)
    else:
        return 'Saturation'