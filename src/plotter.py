# src/plotter.py

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pandas as pd
from scipy.interpolate import interp1d
from helpers import calculate_vth


class Plotter:
    def __init__(self, scaler_X, scaler_y, features_for_model, device):
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.features_for_model = features_for_model
        self.device = device

    def id_vds_characteristics(
        self,
        model,
        full_original_data_for_plot,
        specific_cases_config,
        model_name="Model",
        output_dir="reports/models/final_model_plots/characteristic_plots"
    ):
        os.makedirs(output_dir, exist_ok=True)
        model.eval()

        # Increased tolerance for floating point comparisons of W, L, and Vg
        tolerance_w_l = 1e-6
        tolerance_vg = 0.05

        print(f"\nGenerating Id-Vds plots for {model_name}")
        print(f"Plots will be saved to: {output_dir}")

        for region, cases in specific_cases_config.items():
            print(f"Plotting {region} region")

            #Plot Config
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.set_title(f'{model_name} Performance in {region} Region', fontsize=14)
            ax.set_ylabel('Id (µA)')
            ax.set_xlabel('Vds (V)')
            ax.grid(True, which="major")

            #Id ranges for regions - from the paper
            if region == 'Cut-off':
                ax.set_ylim([0, 12])
            elif region == 'Linear':
                ax.set_ylim([0, 2])
            else:
                ax.set_ylim([0, 1])

            for case in cases:
                label_prefix = case.get('label', 'Case')
                w_val = case.get('W')
                l_val = case.get('L')
                vg_constant = case.get('Vg_const')

                #Set up the x-axis ranges - derived in the main.py from the best/worst calculations
                x_range_min, x_range_max = case.get('Vds_range')
                x_range_dense = np.linspace(x_range_min, x_range_max, 100)

                # Generate synthetic prediction data
                synthetic_data_pred = pd.DataFrame(index=range(len(x_range_dense)))
                synthetic_data_pred['vg'] = vg_constant
                synthetic_data_pred['vd'] = x_range_dense

                #TODO: Vb is not 0.0 in the dataset. Maybe make it the column mean
                synthetic_data_pred['vb'] = 0.0  # Consistent with your model features
                synthetic_data_pred['w'] = w_val
                synthetic_data_pred['l'] = l_val
                synthetic_data_pred['wOverL'] = w_val / l_val
                # 'vgs' and 'vds' are for internal region classification, not model input
                synthetic_data_pred['vgs'] = synthetic_data_pred['vg'] - synthetic_data_pred['vb']
                synthetic_data_pred['vds'] = synthetic_data_pred['vd'] - synthetic_data_pred['vb']

                # Reorder features properly
                X_pred = synthetic_data_pred[self.features_for_model]
                X_pred_scaled = self.scaler_X.transform(X_pred)
                X_tensor = torch.tensor(X_pred_scaled, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    y_pred_scaled_log = model(X_tensor).cpu().numpy()
                y_pred_log = self.scaler_y.inverse_transform(y_pred_scaled_log)
                y_pred = np.power(10, y_pred_log).flatten() * 1e6  # Convert to µA

                ax.plot(x_range_dense, y_pred, '--', label=f'{label_prefix} Predicted', color='red')

                # Filter actual measured data
                # Filter by W and L first with a generous tolerance
                measured_df_filtered = full_original_data_for_plot[
                    (np.isclose(full_original_data_for_plot['w'], w_val, atol=tolerance_w_l)) &
                    (np.isclose(full_original_data_for_plot['l'], l_val, atol=tolerance_w_l))
                    ].copy()

            if not measured_df_filtered.empty:
                # Find the closest 'vg' value in the measured data for the current 'vg_constant'
                unique_vg_vals = measured_df_filtered['vg'].unique()
                if unique_vg_vals.size > 0:
                    closest_vg_val = unique_vg_vals[np.argmin(np.abs(unique_vg_vals - vg_constant))]

                    # Filter for this closest 'vg' and assume 'vb' is also close to 0 if not explicitly 0 in raw data
                    # Given 'vb' is a feature in your model, and you're setting it to 0 for synthetic data,
                    # you should filter measured data where 'vb' is also near 0.
                    # This assumes your original data has 'vb' values that are near 0 for these characteristic plots.
                    measured_subset = measured_df_filtered[
                        (np.isclose(measured_df_filtered['vg'], closest_vg_val, atol=tolerance_vg)) &
                        (np.isclose(measured_df_filtered['vb'], 0.0, atol=tolerance_vg))  # Add vb filter
                        ].copy()

                    if not measured_subset.empty:
                        measured_subset = measured_subset.sort_values(by='vds')
                        # Average Id for same Vds points to smooth out potential noise
                        measured_subset = measured_subset.groupby('vds')['id'].mean().reset_index()

                        if len(measured_subset['vds']) >= 2:
                            # Only interpolate if there are at least two unique Vds points
                            interp_func = interp1d(measured_subset['vds'], measured_subset['id'] * 1e6,
                                                   kind='linear',
                                                   fill_value="extrapolate")  # Consider "extrapolate" carefully
                            measured_id_interp = interp_func(x_range_dense)
                            ax.plot(x_range_dense, measured_id_interp, '-', label=f'{label_prefix} Measured',
                                    color='blue')
                        else:
                            print(
                                f"  Warning: Not enough unique Vds points for interpolation for {label_prefix}. Plotting raw points only.")

                        # Always plot the raw measured points
                        ax.plot(measured_subset['vds'], measured_subset['id'] * 1e6, 'o',
                                markersize=3, color='blue', alpha=0.6)
                    else:
                        print(
                            f"  No measured data found for W={w_val}, L={l_val}, closest Vg={closest_vg_val} (target Vg={vg_constant}) and Vb near 0.")
                else:
                    print(f"  No measured data found for W={w_val}, L={l_val}.")
            ax.legend()

            # Optional debug printouts
            """print("\n[DEBUG]")
            print("Synthetic Data Head (vg, vd, vb, w, l, vgs, vds):")
            print(synthetic_data_pred[['vg', 'vd', 'vb', 'w', 'l', 'vgs', 'vds']].head())
            print("Predicted log_Id (first 5):", y_pred_log[:5].flatten())
            print("Predicted Id µA (first 5):", y_pred[:5])
            if 'measured_subset' in locals() and not measured_subset.empty:
                print("Measured Data Head (vds, id):")
                print(measured_subset[['vds', 'id']].head())
                print(f"Closest Vg value found in measured data: {closest_vg_val}")"""

        print(f"{region} region plot is done.")
        plt.tight_layout()
        plot_filename = f"{model_name.replace(' ', '_').lower()}_{region.lower().replace('-', '_')}.png"
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close(fig)
