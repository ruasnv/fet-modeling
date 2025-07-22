# src/plotter.py

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pandas as pd
from scipy.interpolate import interp1d


class Plotter:
    def __init__(self, scaler_X, scaler_y, features_for_model, device):
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.features_for_model = features_for_model
        self.device = device

    def prepare_model_input_and_predict(
            self, measured_subset, synthetic_data_pred, model, ax, x_range_dense, label_prefix, color_format
    ):
        # Prepare prediction input
        X_pred = synthetic_data_pred[self.features_for_model]
        X_pred_scaled = self.scaler_X.transform(X_pred)
        X_tensor = torch.tensor(X_pred_scaled, dtype=torch.float32).to(self.device)

        # Model prediction
        with torch.no_grad():
            y_pred_scaled = model(X_tensor).cpu().numpy()

        # Inverse transform log_Id
        log_Id_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        # Convert back from log10 to Id
        Id_pred = np.power(10, log_Id_pred)

        # Convert to µA
        y_pred_micro = Id_pred.flatten() * 1e6

        # TODO: TO AVOID NEAR-ZERO NOISE
        y_pred_micro = np.clip(y_pred_micro, 1e-3, None)

        # Plot model prediction (dashed red)
        ax.plot(x_range_dense, y_pred_micro, color_format , label=f'{label_prefix} - Predicted')

        #FOR DEBUGGING
        #print(f"  log_Id_pred (after inverse scaling): {log_Id_pred[:5].flatten()}")
        #print(f"  Id_pred (A): {Id_pred[:5].flatten()}")
        #print(f"  Id_pred (µA): {y_pred_micro[:5]}")

        # Plot interpolated measured data (solid blue)
        grouped = measured_subset.groupby('vd')['id'].mean().reset_index()
        vd_measured = grouped['vd'].values
        id_measured = grouped['id'].values * 1e6  # Convert to µA

        #TODO: TO AVOID NEAR-ZERO NOISE to debug the oscillations
        # Avoid log(0) or log of noise
        id_measured = np.clip(id_measured, 1e-3, None)  # 1e-3 µA = 1e-9 A

        #FOR DEBUGGING
        #print(f"  vd_measured (len={len(vd_measured)}): {vd_measured[:5]}")
        #print(f"  id_measured (µA): {id_measured[:5]}")
        #print(f"  Predicted Id range (scaled): {y_pred_scaled[:5].flatten()}")
        #print(f"  Predicted Id range (unscaled): {y_pred_micro[:5]}")

        if label_prefix == "Best Case":
            color_format_measured  = 'r--'
        else:
            color_format_measured  = 'b--'

        if len(vd_measured) >= 2:

            #TODO: DEBUG
            # Force linear for low-current / noisy regions
            if measured_subset['id'].max() < 1e-9:
                interp_kind = 'linear'
            elif len(vd_measured) >= 4:
                interp_kind = 'cubic'
            else:
                interp_kind = 'linear'

            #interp_kind = 'cubic' if len(vd_measured) >= 4 else 'linear'
            try:
                interp_func = interp1d(vd_measured, id_measured, kind=interp_kind, fill_value="extrapolate")
                id_interpolated = interp_func(x_range_dense)
                ax.plot(x_range_dense, id_interpolated, color_format_measured , label=f'{label_prefix} - Measured')
            except Exception as e:
                print(f"Interpolation failed for {label_prefix} — {str(e)}")
        else:
            print(f"Not enough data points for interpolation in {label_prefix}. Skipping measured plot.")

    def id_vds_characteristics(
            self,
            model,
            full_original_data_for_plot,
            cases_config_for_best_worst_plots,
            model_name="Model",
            output_dir="reports/models/final_model_plots/characteristic_plots"
    ):
        os.makedirs(output_dir, exist_ok=True)
        model.eval()

        print(f"\nGenerating Id–Vds plots for {model_name}")

        for region, cases in cases_config_for_best_worst_plots.items():
            print(f"Plotting {region} region...")

            # Create two figures: linear and log
            fig_linear, ax_linear = plt.subplots(figsize=(10, 7))
            fig_log, ax_log = plt.subplots(figsize=(10, 7))

            ax_linear.set_title(f'{model_name} Performance in {region} Region (Linear Scale)', fontsize=14)
            ax_linear.set_ylabel('Id (µA)')
            ax_linear.set_xlabel('Vds (V)')
            ax_linear.grid(True)

            ax_log.set_title(f'{model_name} Performance in {region} Region (Log Scale)', fontsize=14)
            ax_log.set_ylabel('Id (µA)')
            ax_log.set_xlabel('Vds (V)')
            ax_log.set_yscale('log')
            ax_log.grid(True, which='both', linestyle='--', linewidth=0.5)

            for case in cases:
                label_prefix = case.get('label', 'Case')
                w_val = case.get('W')
                l_val = case.get('L')
                vg_constant = case.get('Vg_const')
                x_range_min, x_range_max = case.get('Vds_range')
                x_range_dense = np.linspace(x_range_min, x_range_max, 100)

                subset = full_original_data_for_plot[
                    (np.isclose(full_original_data_for_plot['w'], w_val, atol=1e-9)) &
                    (np.isclose(full_original_data_for_plot['l'], l_val, atol=1e-9)) &
                    (np.isclose(full_original_data_for_plot['vg'], vg_constant, atol=1e-2))
                    ]

                if subset.empty:
                    print(f"No measured data for {label_prefix}. Skipping...")
                    continue

                vb_mean = subset['vb'].mean()

                synthetic_data_pred = pd.DataFrame({
                    'vg': vg_constant,
                    'vd': x_range_dense,
                    'vb': vb_mean,
                    'w': w_val,
                    'l': l_val
                })

                print(f"\n{label_prefix} Case")
                print(f"  W = {w_val}, L = {l_val}, Vg = {vg_constant}")
                #FOR DEBUGGING:
                #print(f"  Matching measured rows: {len(subset)}")
                #print(f"  Example rows:\n{subset[['vd', 'vg', 'vb', 'id']].head()}")

                if label_prefix == "Best Case":
                    color_format = 'r-'
                else:
                    color_format = 'b-'

                # Plot both linear and log-scale
                self.prepare_model_input_and_predict(
                    measured_subset=subset,
                    synthetic_data_pred=synthetic_data_pred,
                    model=model,
                    ax=ax_linear,
                    x_range_dense=x_range_dense,
                    label_prefix=label_prefix,
                    color_format = color_format
                )
                self.prepare_model_input_and_predict(
                    measured_subset=subset,
                    synthetic_data_pred=synthetic_data_pred,
                    model=model,
                    ax=ax_log,
                    x_range_dense=x_range_dense,
                    label_prefix=label_prefix,
                    color_format =  color_format
                )

            #FOR DEBUGGING:
            # print(np.log10(full_original_data_for_plot['id']).describe())

            ax_linear.legend()
            ax_log.legend()
            plt.tight_layout()

            # Save both figures
            filename_base = f"{model_name.replace(' ', '_').lower()}_{region.lower().replace('-', '_')}"
            linear_path = os.path.join(output_dir, f"{filename_base}_linear.png")
            log_path = os.path.join(output_dir, f"{filename_base}_log.png")

            fig_linear.savefig(linear_path)
            fig_log.savefig(log_path)
            plt.close(fig_linear)
            plt.close(fig_log)

            print(f"Finished linear plot for {region} region → saved: {linear_path}")
            print(f"Finished log-scale plot for {region} region → saved: {log_path}")
