# src/utils/plotter.py

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path
from src.config import settings # Import settings for global plot parameters

class Plotter:
    def __init__(self, scaler_X, scaler_y, features_for_model, device):
        """
        Initializes the Plotter with necessary scaling and device information.

        Args:
            scaler_X (sklearn.preprocessing.StandardScaler): Scaler fitted on input features (X).
            scaler_y (sklearn.preprocessing.StandardScaler): Scaler fitted on target feature (y, log_Id).
            features_for_model (list): List of feature names used by the model.
            device (torch.device): The device (CPU or CUDA) where the model performs inference.
        """
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.features_for_model = features_for_model
        self.device = device

    def prepare_model_input_and_predict(
            self, measured_subset, model, ax, x_range_dense, case_label, measured_color, predicted_color, region_name
    ):
        """
        Generates synthetic input data for prediction, runs the model, inverse transforms results,
        and plots both predicted and measured data on a given Matplotlib axis.

        Args:
            measured_subset (pd.DataFrame): The subset of original measured data for the current case.
            model (torch.nn.Module): The trained neural network model.
            ax (matplotlib.axes.Axes): The Matplotlib axes to plot on.
            x_range_dense (np.ndarray): A dense array of Vds values for smooth plotting.
            case_label (str): A detailed label for the specific case (e.g., "W=1um, L=0.5um, Vg=1.5V").
            measured_color (str): Matplotlib color string for measured data.
            predicted_color (str): Matplotlib color string for predicted data.
            region_name (str): The name of the operating region (e.g., "Cut-off", "Linear", "Saturation").
        """
        # Ensure model is in evaluation mode
        model.eval()

        # Extract W, L, Vg, Vb (mean for vb) from the measured subset for synthetic data generation
        # Assuming these are constant for the specific case
        w_val = measured_subset['w'].iloc[0]
        l_val = measured_subset['l'].iloc[0]
        vg_constant = measured_subset['vg'].iloc[0]
        # Use the specific Vbs value from the config if available, otherwise mean from data
        # This assumes plot_cases config will contain a 'vbs_val' for each case
        vb_val = measured_subset['vb'].iloc[0] # Assuming vb is constant for a given case in measured data

        # Create synthetic data points for prediction across the dense Vds range
        synthetic_data_pred = pd.DataFrame({
            'vg': vg_constant,
            'vd': x_range_dense,
            'vb': vb_val, # Use the specific vb_val
            'w': w_val,
            'l': l_val
        })

        # Check if 'wOverL' is an expected feature and add it if so
        synthetic_data_pred['wOverL'] = synthetic_data_pred['w'] / synthetic_data_pred['l']

        # Prepare prediction input: select features and scale them
        X_pred = synthetic_data_pred[self.features_for_model]
        X_pred_scaled = self.scaler_X.transform(X_pred)
        X_tensor = torch.tensor(X_pred_scaled, dtype=torch.float32).to(self.device)

        # Model prediction
        with torch.no_grad():
            y_pred_scaled = model(X_tensor).cpu().numpy()

        # Inverse transform log_Id predictions back to original Id scale (Amperes)
        log_Id_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        Id_pred = np.power(10, log_Id_pred)  # Convert from log10(Id) to Id (Amperes)

        # Convert to microamperes (µA) for plotting and clip near-zero noise
        y_pred_micro = Id_pred.flatten() * 1e6
        # Clip very small values to a small positive to avoid issues on log plots (1e-6 µA = 1e-12 A, i.e., pA)
        y_pred_micro = np.clip(y_pred_micro, 1e-6, None)

        # Plot model prediction
        ax.plot(x_range_dense, y_pred_micro, predicted_color, linestyle='--', label=f'Predicted ({case_label})')

        # Plot interpolated measured data
        grouped_measured = measured_subset.groupby('vd')['id'].mean().reset_index()
        vd_measured = grouped_measured['vd'].values
        id_measured = grouped_measured['id'].values * 1e6  # Convert to µA

        # Clip measured Id values to avoid issues with log scale or near-zero noise
        id_measured = np.clip(id_measured, 1e-6, None)

        if len(vd_measured) >= 2:
            interp_kind = 'cubic' if len(vd_measured) >= 4 else 'linear'
            if np.max(id_measured) < 1e-3: # If max current is less than 1 nA (0.001 µA)
                interp_kind = 'linear'

            try:
                # Create interpolation function. Use 'extrapolate' to cover the full x_range_dense.
                interp_func = interp1d(vd_measured, id_measured, kind=interp_kind, fill_value="extrapolate")
                id_interpolated = interp_func(x_range_dense)
                ax.plot(x_range_dense, id_interpolated, measured_color, linestyle='-', label=f'Measured ({case_label})')
            except Exception as e:
                print(
                    f"  Warning: Interpolation failed for {case_label} in {region_name} region. Error: {str(e)}. Skipping measured plot for this case.")
        else:
            print(
                f"  Not enough data points ({len(vd_measured)}) for interpolation in {case_label} in {region_name} region. Skipping measured plot for this case.")

    def id_vds_characteristics(
            self,
            model,
            full_original_data_for_plot,
            cases_config_for_best_worst_plots,
            model_name="Model",
            output_dir: Path = None # Ensure output_dir is a Path object
    ):
        """
        Generates and saves Id-Vds characteristic plots for specified best/worst cases
        in different operating regions. Plots are generated on both linear and log scales.

        Args:
            model (torch.nn.Module): The trained neural network model.
            full_original_data_for_plot (pd.DataFrame): The full preprocessed DataFrame
                including original 'id' values, 'w', 'l', 'vg', 'vd', 'vb', 'operating_region'.
            cases_config_for_best_worst_plots (dict): A dictionary defining the
                W, L, Vg combinations for plotting, typically from `determine_characteristic_plot_cases`.
            model_name (str): Name of the model for plot titles and filenames.
            output_dir (Path): Directory where the generated plots will be saved.
        """
        if output_dir is None:
            output_dir = Path("results/plots/characteristic_plots") # Default if not provided
        os.makedirs(output_dir, exist_ok=True)
        model.eval()  # Set model to evaluation mode once for all predictions

        print(f"\nGenerating Id–Vds characteristic plots for {model_name}")

        # Define consistent colors for best/worst cases across plots
        color_best_measured = 'blue'
        color_best_predicted = 'cornflowerblue'
        color_worst_measured = 'red'
        color_worst_predicted = 'lightcoral'

        for region, cases in cases_config_for_best_worst_plots.items():
            print(f"Plotting {region} region...")

            fig_linear, ax_linear = plt.subplots(figsize=settings.get('global_settings.figure_figsize'))
            fig_log, ax_log = plt.subplots(figsize=settings.get('global_settings.figure_figsize'))

            # Set common plot properties for linear scale
            ax_linear.set_title(f'{model_name} Performance in {region} Region (Linear Scale)',
                                fontsize=settings.get('global_settings.axes_titlesize'))
            ax_linear.set_ylabel('Id (µA)', fontsize=settings.get('global_settings.axes_labelsize'))
            ax_linear.set_xlabel('Vds (V)', fontsize=settings.get('global_settings.axes_labelsize'))
            ax_linear.grid(True)

            # Set common plot properties for log scale
            ax_log.set_title(f'{model_name} Performance in {region} Region (Log Scale)',
                             fontsize=settings.get('global_settings.axes_titlesize'))
            ax_log.set_ylabel('Id (µA)', fontsize=settings.get('global_settings.axes_labelsize'))
            ax_log.set_xlabel('Vds (V)', fontsize=settings.get('global_settings.axes_labelsize'))
            ax_log.set_yscale('log')
            ax_log.grid(True, which='both', linestyle='--', linewidth=0.5)

            for case_idx, case in enumerate(cases):
                # FIX: Create a more informative label
                w_val_um = case.get('device_size')[0]
                l_val_um = case.get('device_size')[1]
                vbs_val = case.get('vbs_val')
                vg_const = case.get('fixed_vgs_vals', [None])[0] # Assuming fixed_vgs_vals if Id_Vds plot
                vd_const = case.get('fixed_vds_vals', [None])[0] # Assuming fixed_vds_vals if Id_Vgs plot

                case_label = f"W={w_val_um:.1f}µm, L={l_val_um:.1f}µm, Vbs={vbs_val:.1f}V"
                if vg_const is not None:
                    case_label += f", Vgs={vg_const:.1f}V"
                if vd_const is not None:
                    case_label += f", Vds={vd_const:.1f}V"

                # Extract Vds_range or Vgs_range based on plot_type
                if case.get('plot_type') == "Id_Vds_fixed_Vgs":
                    x_range_min, x_range_max = case.get('Vds_range', [0.0, 3.3]) # Default Vds range
                    x_range_dense = np.linspace(x_range_min, x_range_max, 100)
                    vg_filter_val = case.get('fixed_vgs_vals')[0] # Assuming only one Vgs for this plot type
                    vd_filter_val = None # Not fixed Vds for this plot type
                    # Use Vgs for filtering
                    vg_to_filter = vg_filter_val
                    vd_to_filter = None
                    x_axis_col = 'vd'
                elif case.get('plot_type') == "Id_Vgs_fixed_Vds":
                    x_range_min, x_range_max = case.get('Vgs_range', [0.0, 3.3]) # Default Vgs range
                    x_range_dense = np.linspace(x_range_min, x_range_max, 100)
                    vd_filter_val = case.get('fixed_vds_vals')[0] # Assuming only one Vds for this plot type
                    vg_filter_val = None # Not fixed Vgs for this plot type
                    # Use Vds for filtering
                    vg_to_filter = None
                    vd_to_filter = vd_filter_val
                    x_axis_col = 'vg'
                else:
                    print(f"  Warning: Unsupported plot_type '{case.get('plot_type')}' for case {case_label}. Skipping.")
                    continue


                # Filter the original data to get the measured subset for this specific case
                measured_subset = full_original_data_for_plot[
                    (np.isclose(full_original_data_for_plot['w'], w_val_um * 1e-6, atol=1e-9)) & # Convert um to meters
                    (np.isclose(full_original_data_for_plot['l'], l_val_um * 1e-6, atol=1e-9)) & # Convert um to meters
                    (np.isclose(full_original_data_for_plot['vb'], vbs_val, atol=1e-2))
                ].copy()

                # Apply specific Vgs or Vds filter based on plot type
                if vg_to_filter is not None:
                    measured_subset = measured_subset[np.isclose(measured_subset['vg'], vg_to_filter, atol=1e-2)].copy()
                if vd_to_filter is not None:
                    measured_subset = measured_subset[np.isclose(measured_subset['vd'], vd_to_filter, atol=1e-2)].copy()


                if measured_subset.empty:
                    print(f"  No measured data found for {case_label} in {region} region. Skipping this case.")
                    continue

                print(f"  Plotting case: {case_label}")

                # Assign colors based on "Best Case" or "Worst Case"
                if "best_case" in case.get('label', '').lower(): # Check label from config
                    current_measured_color = color_best_measured
                    current_predicted_color = color_best_predicted
                elif "worst_case" in case.get('label', '').lower(): # Check label from config
                    current_measured_color = color_worst_measured
                    current_predicted_color = color_worst_predicted
                else: # Fallback for other cases
                    current_measured_color = 'purple'
                    current_predicted_color = 'plum'

                # Call prepare_model_input_and_predict
                self.prepare_model_input_and_predict(
                    measured_subset=measured_subset,
                    model=model,
                    ax=ax_linear,
                    x_range_dense=x_range_dense,
                    case_label=case_label, # Pass the detailed case label
                    measured_color=current_measured_color,
                    predicted_color=current_predicted_color,
                    region_name=region
                )
                self.prepare_model_input_and_predict(
                    measured_subset=measured_subset,
                    model=model,
                    ax=ax_log,
                    x_range_dense=x_range_dense,
                    case_label=case_label, # Pass the detailed case label
                    measured_color=current_measured_color,
                    predicted_color=current_predicted_color,
                    region_name=region
                )

            # Add legends and save figures
            ax_linear.legend(fontsize=settings.get('global_settings.legend_fontsize'))
            ax_log.legend(fontsize=settings.get('global_settings.legend_fontsize'))
            plt.tight_layout()

            # Save both figures
            filename_base = f"{model_name.replace(' ', '_').lower()}_{region.lower().replace('-', '_')}_characteristics"
            linear_path = output_dir / f"{filename_base}_linear.png"
            log_path = output_dir / f"{filename_base}_log.png"

            fig_linear.savefig(linear_path)
            fig_log.savefig(log_path)
            plt.close(fig_linear)
            plt.close(fig_log)

            print(f"  Saved linear plot for {region} region → {linear_path}")
            print(f"  Saved log-scale plot for {region} region → {log_path}")

