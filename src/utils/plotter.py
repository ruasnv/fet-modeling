# src/utils/plotter.py

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path
from src.config import settings


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
            self, measured_subset, model, ax, x_range_dense, case_label, measured_color, predicted_color, region_name,
            fixed_val_for_synthetic, sweep_axis_col
    ):
        """
        Generates synthetic input data for prediction, runs the model, inverse transforms results,
        and plots both predicted and measured data on a given Matplotlib axis.
        """
        model.eval()

        w_val = measured_subset['w'].iloc[0]
        l_val = measured_subset['l'].iloc[0]
        vb_val = measured_subset['vb'].iloc[0]

        synthetic_data_dict = {
            'w': w_val,
            'l': l_val,
            'vb': vb_val
        }

        if sweep_axis_col == 'vd':
            synthetic_data_dict['vd'] = np.full_like(x_range_dense, x_range_dense)
            synthetic_data_dict['vg'] = np.full_like(x_range_dense, fixed_val_for_synthetic)
        elif sweep_axis_col == 'vg':
            synthetic_data_dict['vg'] = np.full_like(x_range_dense, x_range_dense)
            synthetic_data_dict['vd'] = np.full_like(x_range_dense, fixed_val_for_synthetic)
        else:
            raise ValueError(f"Unsupported sweep_axis_col: {sweep_axis_col}")

        synthetic_data_pred = pd.DataFrame(synthetic_data_dict)
        synthetic_data_pred['wOverL'] = synthetic_data_pred['w'] / synthetic_data_pred['l']

        X_pred = synthetic_data_pred[self.features_for_model]
        X_pred_scaled = self.scaler_X.transform(X_pred)
        X_tensor = torch.tensor(X_pred_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred_scaled = model(X_tensor).cpu().numpy()

        log_Id_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        Id_pred = np.power(10, log_Id_pred)

        y_pred_micro = Id_pred.flatten() * 1e6
        y_pred_micro = np.clip(y_pred_micro, 1e-6, None)

        ax.plot(x_range_dense, y_pred_micro, predicted_color, linestyle='--', label=f'Predicted ({case_label})')

        grouped_measured = measured_subset.groupby(sweep_axis_col)['id'].mean().reset_index()
        x_measured = grouped_measured[sweep_axis_col].values
        id_measured = grouped_measured['id'].values * 1e6

        id_measured = np.clip(id_measured, 1e-6, None)

        if len(x_measured) >= 2:
            # Create a separate, dense range for the measured data to prevent extrapolation
            x_measured_dense = np.linspace(x_measured.min(), x_measured.max(), 100)

            interp_kind = 'cubic' if len(x_measured) >= 4 else 'linear'
            if np.max(id_measured) < 1e-3:
                interp_kind = 'linear'

            try:
                # Interpolate and plot on the measured data's own dense range
                interp_func = interp1d(x_measured, id_measured, kind=interp_kind, fill_value="extrapolate")
                id_interpolated = interp_func(x_measured_dense)
                ax.plot(x_measured_dense, id_interpolated, measured_color, linestyle='-',
                        label=f'Measured ({case_label})')
            except Exception as e:
                print(
                    f"  Warning: Interpolation failed for {case_label} in {region_name}. Error: {str(e)}. Skipping measured plot for this case.")
        else:
            print(
                f"  Not enough data points ({len(x_measured)}) for interpolation in {case_label} in {region_name}. Skipping measured plot for this case.")


    def id_vds_characteristics(
            self,
            model,
            full_original_data_for_plot,
            cases_config_for_best_worst_plots: dict,
            model_name="Model",
            output_dir: Path = None
    ):
        """
        Generates and saves Id-Vds characteristic plots for specified best/worst cases
        in different operating regions. Plots are generated on both linear and log scales.

        Args:
            model (torch.nn.Module): The trained neural network model.
            full_original_data_for_plot (pd.DataFrame): The full preprocessed DataFrame
                including original 'id' values, 'w', 'l', 'vg', 'vd', 'vb', 'operating_region'.
            cases_config_for_best_worst_plots (dict): A dictionary where keys are operating regions
                (e.g., "Cut-off") and values are lists of case dictionaries (e.g., [best_case, worst_case]).
            model_name (str): Name of the model for plot titles and filenames.
            output_dir (Path): Directory where the generated plots will be saved.
        """
        if output_dir is None:
            output_dir = Path("results/plots/characteristic_plots")
        os.makedirs(output_dir, exist_ok=True)
        model.eval()

        print(f"\nGenerating Id characteristic plots for {model_name}")

        # Colors for best/worst case plots
        color_best_measured = 'blue'
        color_best_predicted = 'cornflowerblue'
        color_worst_measured = 'red'
        color_worst_predicted = 'lightcoral'

        #Iterate over regions and their associated cases
        for region, cases_list in cases_config_for_best_worst_plots.items():
            print(f"Plotting for {region} region...")

            # Create new figures for each region (to group best/worst on one plot)
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

            # Iterate through best_case and worst_case for the current region
            for case_dict in cases_list:
                # Extract parameters from the dynamically generated case_dict
                w_val_um = case_dict.get('device_size')[0]
                l_val_um = case_dict.get('device_size')[1]
                vbs_val = case_dict.get('vbs_val')
                vg_const = case_dict.get('fixed_vgs_vals')[0]  # Vg_const from suggest_best_worst_cases
                vds_range = case_dict.get('Vds_range')
                case_label = case_dict.get('label')  # Use the label generated by suggest_best_worst_cases

                # Determine x_range_dense and sweep_axis_col
                x_range_min, x_range_max = vds_range
                x_range_dense = np.linspace(x_range_min, x_range_max, 100)
                sweep_axis_col = 'vd'

                # Determine fixed value for synthetic data generation (Vg_const for Id-Vds)
                fixed_val_for_synthetic = vg_const

                # Filter the original data to get the measured subset for this specific case
                measured_subset = full_original_data_for_plot[
                    (np.isclose(full_original_data_for_plot['w'], w_val_um * 1e-6, atol=1e-9)) &
                    (np.isclose(full_original_data_for_plot['l'], l_val_um * 1e-6, atol=1e-9)) &
                    (np.isclose(full_original_data_for_plot['vb'], vbs_val, atol=1e-2)) &
                    (np.isclose(full_original_data_for_plot['vg'], vg_const, atol=1e-2))
                    ].copy()

                if measured_subset.empty:
                    print(f"  No measured data found for {case_label} in {region} region. Skipping this line.")
                    continue

                print(f"  Plotting case: {case_label}")

                # Assign colors
                if "best case" in case_label.lower():
                    current_measured_color = color_best_measured
                    current_predicted_color = color_best_predicted
                elif "worst case" in case_label.lower():
                    current_measured_color = color_worst_measured
                    current_predicted_color = color_worst_predicted
                else:
                    current_measured_color = 'purple'
                    current_predicted_color = 'plum'

                # Call prepare_model_input_and_predict for both linear and log axes
                self.prepare_model_input_and_predict(
                    measured_subset=measured_subset,
                    model=model,
                    ax=ax_linear,
                    x_range_dense=x_range_dense,
                    case_label=case_label,
                    measured_color=current_measured_color,
                    predicted_color=current_predicted_color,
                    region_name=region,
                    fixed_val_for_synthetic=fixed_val_for_synthetic,
                    sweep_axis_col=sweep_axis_col
                )
                self.prepare_model_input_and_predict(
                    measured_subset=measured_subset,
                    model=model,
                    ax=ax_log,
                    x_range_dense=x_range_dense,
                    case_label=case_label,
                    measured_color=current_measured_color,
                    predicted_color=current_predicted_color,
                    region_name=region,
                    fixed_val_for_synthetic=fixed_val_for_synthetic,
                    sweep_axis_col=sweep_axis_col
                )

            # Add legends and save figures for specific region's plots
            ax_linear.legend(fontsize=settings.get('global_settings.legend_fontsize'))
            ax_log.legend(fontsize=settings.get('global_settings.legend_fontsize'))
            plt.tight_layout()

            # Filename
            filename_base = f"{model_name.replace(' ', '_').lower()}_{region.lower().replace('-', '_')}_characteristics"
            linear_path = output_dir / f"{filename_base}_linear.png"
            log_path = output_dir / f"{filename_base}_log.png"

            fig_linear.savefig(linear_path)
            fig_log.savefig(log_path)
            plt.close(fig_linear)
            plt.close(fig_log)

            print(f"  Saved linear plot for {region} region → {linear_path}")
            print(f"  Saved log-scale plot for {region} region → {log_path}")
