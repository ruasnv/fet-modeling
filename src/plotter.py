# src/plotter.py

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pandas as pd
from scipy.interpolate import interp1d  # Import for interpolation


class Plotter:
    def __init__(self, scaler_X, scaler_y, features_for_model, device):
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.features_for_model = features_for_model
        self.device = device  # Needed for PyTorch model predictions

    def plot_fet_characteristics(
            self,
            model,  # The trained model instance (PyTorch nn.Module)
            full_original_data_for_plot,  # Now takes the full filtered original data
            specific_cases_config,  # Now takes specific config
            model_name="Model",
            output_dir="reports/models/final_model_plots/characteristic_plots"  # More general name
    ):
        """
        Plots Id vs Vds and Id vs Vgs curves for specific best-case and worst-case scenarios,
        comparing model predictions to measurements, and saves them.
        Args:
            model: The trained PyTorch model instance.
            full_original_data_for_plot (pd.DataFrame): The full original DataFrame after initial filtering and preprocessing.
            specific_cases_config (dict): A dictionary defining the specific W, L, and constant
                                          Vg/Vds values for best/worst cases in each region.
                                          Expected structure:
                                          {
                                              'Region': {
                                                  'id_vs_vds': [
                                                      {'label': 'Case 1', 'W': ..., 'L': ..., 'Vg_const': ..., 'Vds_range': [min, max]},
                                                      {'label': 'Case 2', 'W': ..., 'L': ..., 'Vg_const': ..., 'Vds_range': [min, max]}
                                                  ],
                                                  'id_vs_vgs': [
                                                      {'label': 'Case 1', 'W': ..., 'L': ..., 'Vds_const': ..., 'Vgs_range': [min, max]}
                                                  ]
                                              }
                                          }
            model_name (str): Name of the model being plotted.
            output_dir (str): Directory where plots will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n--- Generating specific FET characteristic plots for {model_name} ---")
        print(f"Plots will be saved to: {output_dir}")

        model.eval()  # Set model to evaluation mode
        tolerance = 1e-7  # Tolerance for floating point comparisons

        # Iterate through each operating region defined in the config
        for region, plot_types in specific_cases_config.items():
            print(f"\n--- Plotting for {region} Region ---")

            # Iterate through plot types (id_vs_vds, id_vs_vgs)
            for plot_type, cases_list in plot_types.items():
                if not cases_list:
                    print(f"No cases defined for {plot_type} in {region}. Skipping.")
                    continue

                fig, ax = plt.subplots(figsize=(10, 7))  # Single subplot per plot_type
                ax.set_title(f'{model_name} Performance in {region} Region: {plot_type.replace("_", " ").upper()}',
                             fontsize=14)
                ax.set_ylabel('Id (µA)')  # Changed to microamperes
                ax.grid(True, which="both" if plot_type == 'id_vs_vgs' else "major")  # Grid for log scale too

                if plot_type == 'id_vs_vds':
                    varying_col = 'vds'
                    constant_col = 'vgs'
                    x_label = 'Vds (V)'
                    ax.set_xlabel(x_label)
                    # Specific Y-axis limits for Cut-off Vds plot as per paper Fig 3
                    if region == 'Cut-off':
                        ax.set_ylim([0, 12])  # Id in muA
                elif plot_type == 'id_vs_vgs':
                    varying_col = 'vgs'
                    constant_col = 'vds'
                    x_label = 'Vgs (V)'
                    ax.set_xlabel(x_label)
                    ax.set_yscale('log')  # Id vs Vgs often better on log scale
                else:
                    print(f"Unknown plot type: {plot_type}. Skipping.")
                    plt.close(fig)  # Close the empty figure
                    continue

                # Store all Id values to determine overall min/max for dynamic y-limits if not fixed
                all_ids_for_ylim = []

                # Iterate through each specific case (e.g., best, worst) to plot on the same graph
                for case_data in cases_list:
                    label_prefix = case_data.get('label', 'Case')
                    w_val = case_data.get('W')
                    l_val = case_data.get('L')

                    if plot_type == 'id_vs_vds':
                        const_val = case_data.get('Vg_const')
                        x_range = np.linspace(case_data.get('Vds_range')[0], case_data.get('Vds_range')[1],
                                              100)  # 100 points for smooth curve
                    else:  # id_vs_vgs
                        const_val = case_data.get('Vds_const')
                        x_range = np.linspace(case_data.get('Vgs_range')[0], case_data.get('Vgs_range')[1],
                                              100)  # 100 points for smooth curve

                    # --- Generate Predicted Curve ---
                    # Create a DataFrame for synthetic input points
                    synthetic_data = pd.DataFrame(index=range(len(x_range)), columns=self.features_for_model)
                    synthetic_data['w'] = w_val
                    synthetic_data['l'] = l_val
                    synthetic_data['vb'] = 0.0  # Assuming Vb=0 for these plots, adjust if needed
                    synthetic_data['wOverL'] = w_val / l_val

                    if plot_type == 'id_vs_vds':
                        synthetic_data['vds'] = x_range
                        synthetic_data['vg'] = const_val  # Vg_const
                        synthetic_data['vgs'] = synthetic_data['vg'] - synthetic_data['vb']  # Assuming Vs=Vb=0
                    else:  # id_vs_vgs
                        synthetic_data['vgs'] = x_range
                        synthetic_data['vd'] = const_val  # Vds_const
                        synthetic_data['vg'] = synthetic_data['vgs'] + synthetic_data['vb']  # Assuming Vs=Vb=0
                        synthetic_data['vds'] = synthetic_data['vd'] - synthetic_data['vb']  # Assuming Vs=Vb=0

                    # Ensure all features_for_model are present in synthetic_data
                    # Fill any missing columns (e.g., 'temp' if it's a feature but not varied here)
                    for feature in self.features_for_model:
                        if feature not in synthetic_data.columns:
                            # Assuming a default value like 27.0 for temperature if it's a feature
                            # You might need to adjust this default based on your data's typical values
                            synthetic_data[feature] = 27.0

                    X_pred_scaled = self.scaler_X.transform(synthetic_data[self.features_for_model])
                    X_pred_tensor = torch.tensor(X_pred_scaled, dtype=torch.float32).to(self.device)

                    with torch.no_grad():
                        y_pred_scaled_log = model(X_pred_tensor).cpu().numpy()
                    y_pred_id = np.power(10, self.scaler_y.inverse_transform(
                        y_pred_scaled_log)).flatten() * 1e6  # Convert to µA

                    ax.plot(x_range, y_pred_id, '--', label=f'{label_prefix} Predicted',
                            color='red')  # Dashed for predicted
                    all_ids_for_ylim.extend(y_pred_id)

                    # --- Overlay Measured Data Points (from full_original_data_for_plot) ---
                    # Filter original data for this specific W, L
                    measured_df_filtered_wl = full_original_data_for_plot[
                        (np.isclose(full_original_data_for_plot['w'], w_val, atol=tolerance)) &
                        (np.isclose(full_original_data_for_plot['l'], l_val, atol=tolerance))
                        ].copy()

                    if not measured_df_filtered_wl.empty:
                        # Find the closest constant value in the *measured* data for this W,L
                        unique_measured_const_vals = measured_df_filtered_wl[constant_col].unique()
                        if unique_measured_const_vals.size > 0:
                            closest_measured_const_val = unique_measured_const_vals[
                                np.argmin(np.abs(unique_measured_const_vals - const_val))]

                            measured_subset = measured_df_filtered_wl[
                                np.isclose(measured_df_filtered_wl[constant_col], closest_measured_const_val,
                                           atol=tolerance)
                            ].sort_values(by=varying_col)

                            if not measured_subset.empty:
                                measured_id_muA = measured_subset['id'].values * 1e6

                                # --- Interpolate measured data for a smooth line ---
                                # Ensure enough points for interpolation (at least 2)
                                if len(measured_subset[varying_col]) >= 2:
                                    # Create interpolation function
                                    interp_func = interp1d(measured_subset[varying_col], measured_id_muA, kind='linear',
                                                           fill_value="extrapolate")
                                    # Generate interpolated Id values over the same x_range as predictions
                                    interpolated_measured_id_muA = interp_func(x_range)
                                    ax.plot(x_range, interpolated_measured_id_muA, '-',
                                            label=f'{label_prefix} Measured',
                                            color='blue')  # Solid line for interpolated measured
                                else:
                                    print(
                                        f"Warning: Not enough measured points ({len(measured_subset[varying_col])}) for interpolation for {label_prefix} (W={w_val * 1e6:.1f}µm, L={l_val * 1e6:.1f}µm, {constant_col}={const_val:.2f}V). Plotting discrete points only.")

                                # Always plot original discrete points as markers
                                ax.plot(measured_subset[varying_col], measured_id_muA, 'o', markersize=3, color='blue',
                                        alpha=0.6)  # Original points
                                all_ids_for_ylim.extend(measured_id_muA)
                            else:
                                print(
                                    f"Warning: No measured data found in full original data for {label_prefix} (W={w_val * 1e6:.1f}µm, L={l_val * 1e6:.1f}µm, {constant_col}={const_val:.2f}V) after finding closest constant value.")
                        else:
                            print(
                                f"Warning: No unique constant values found in full original data for {label_prefix} (W={w_val * 1e6:.1f}µm, L={l_val * 1e6:.1f}µm).")
                    else:
                        print(
                            f"Warning: No W={w_val * 1e6:.1f}µm L={l_val * 1e6:.1f}µm data found in full original data for {label_prefix}.")

                ax.legend()

                # Dynamic Y-axis limits if not fixed by config (e.g., Cut-off region)
                if region != 'Cut-off' or plot_type != 'id_vs_vds':  # Apply dynamic limits unless it's the specific cutoff case
                    if all_ids_for_ylim:
                        min_id_val = min(all_ids_for_ylim)
                        max_id_val = max(all_ids_for_ylim)
                        if min_id_val > 0:
                            ax.set_ylim([max(1e-15, min_id_val * 0.5), max_id_val * 1.5])
                        else:
                            ax.set_ylim([1e-15, None])  # Set a floor for log scale if min is 0 or negative

                plt.tight_layout()  # Adjust layout to prevent labels overlapping
                plot_filename = f"{model_name.replace(' ', '_').lower()}_{region.replace('-', '_').lower()}_{plot_type}.png".replace(
                    '.', 'p').replace('-', 'm')
                plt.savefig(os.path.join(output_dir, plot_filename))
                plt.close(fig)  # Close the figure to free memory

