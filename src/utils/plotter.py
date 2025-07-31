# src/utils/plotter.py

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pandas as pd
from scipy.interpolate import interp1d


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
            self, measured_subset, model, ax, x_range_dense, label_prefix, measured_color, predicted_color
    ):
        """
        Prepares synthetic input data for prediction, runs the model, inverse transforms results,
        and plots both predicted and measured data on a given Matplotlib axis.

        Args:
            measured_subset (pd.DataFrame): The subset of original measured data for the current case.
            model (torch.nn.Module): The trained neural network model.
            ax (matplotlib.axes.Axes): The Matplotlib axes to plot on.
            x_range_dense (np.ndarray): A dense array of Vds values for smooth plotting.
            label_prefix (str): Label for the legend (e.g., "Best Case", "Worst Case").
            measured_color (str): Matplotlib color string for measured data.
            predicted_color (str): Matplotlib color string for predicted data.
        """
        # Ensure model is in evaluation mode
        model.eval()

        # Extract W, L, Vg, Vb (mean for vb) from the measured subset for synthetic data generation
        # Assuming these are constant for the specific case
        w_val = measured_subset['w'].iloc[0]
        l_val = measured_subset['l'].iloc[0]
        vg_constant = measured_subset['vg'].iloc[0]
        vb_mean = measured_subset['vb'].mean()  # Using mean vb, as vb might vary slightly or be constant

        # Create synthetic data points for prediction across the dense Vds range
        synthetic_data_pred = pd.DataFrame({
            'vg': vg_constant,
            'vd': x_range_dense,
            'vb': vb_mean,
            'w': w_val,
            'l': l_val
        })

        # Check if 'wOverL' is an expected feature and add it if so
        if 'wOverL' in self.features_for_model:
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
        y_pred_micro = np.clip(y_pred_micro, 1e-6,
                               None)  # Clip very small values to a small positive to avoid log(0) issues on log plots. (1e-6 µA = 1e-12 A, i.e., pA)

        # Plot model prediction
        ax.plot(x_range_dense, y_pred_micro, predicted_color, label=f'{label_prefix} - Predicted')

        # FOR DEBUGGING:
        # print(f"  log_Id_pred (after inverse scaling): {log_Id_pred[:5].flatten()}")
        # print(f"  Id_pred (A): {Id_pred[:5].flatten()}")
        # print(f"  Id_pred (µA): {y_pred_micro[:5]}")

        # Plot interpolated measured data
        # Group measured data by Vd to get average Id if multiple readings exist for same Vd
        grouped_measured = measured_subset.groupby('vd')['id'].mean().reset_index()
        vd_measured = grouped_measured['vd'].values
        id_measured = grouped_measured['id'].values * 1e6  # Convert to µA

        # Clip measured Id values to avoid issues with log scale or near-zero noise
        id_measured = np.clip(id_measured, 1e-6, None)  # Same clipping as for predicted values

        # FOR DEBUGGING:
        # print(f"  vd_measured (len={len(vd_measured)}): {vd_measured[:5]}")
        # print(f"  id_measured (µA): {id_measured[:5]}")

        # Perform interpolation only if enough data points exist
        if len(vd_measured) >= 2:
            # Determine interpolation kind: cubic for smoother curves if enough points, otherwise linear
            interp_kind = 'cubic' if len(vd_measured) >= 4 else 'linear'

            # Add a heuristic: if max current is very low, use linear to avoid cubic oscillations
            if np.max(id_measured) < 1e-3:  # If max current is less than 1 nA (0.001 µA)
                interp_kind = 'linear'

            try:
                # Create interpolation function. Use 'extrapolate' to cover the full x_range_dense.
                interp_func = interp1d(vd_measured, id_measured, kind=interp_kind, fill_value="extrapolate")
                id_interpolated = interp_func(x_range_dense)
                ax.plot(x_range_dense, id_interpolated, measured_color, label=f'{label_prefix} - Measured')
            except Exception as e:
                print(
                    f"  Warning: Interpolation failed for {label_prefix} ({region}). Error: {str(e)}. Skipping measured plot for this case.")
        else:
            print(
                f"  Not enough data points ({len(vd_measured)}) for interpolation in {label_prefix} ({region}). Skipping measured plot for this case.")

    def id_vds_characteristics(
            self,
            model,
            full_original_data_for_plot,
            cases_config_for_best_worst_plots,
            model_name="Model",
            output_dir="results/plots/characteristic_plots"  # Default if not provided
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
            output_dir (str): Directory where the generated plots will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
        model.eval()  # Set model to evaluation mode once for all predictions

        print(f"\nGenerating Id–Vds characteristic plots for {model_name}")

        # Define consistent colors for best/worst cases across plots
        color_best_measured = 'blue'  # Solid blue for best measured
        color_best_predicted = 'cornflowerblue'  # Dashed blue for best predicted
        color_worst_measured = 'red'  # Solid red for worst measured
        color_worst_predicted = 'lightcoral'  # Dashed red for worst predicted

        for region, cases in cases_config_for_best_worst_plots.items():
            print(f"Plotting {region} region...")

            # Create two figures for linear and log scale plots
            fig_linear, ax_linear = plt.subplots(figsize=(10, 7))
            fig_log, ax_log = plt.subplots(figsize=(10, 7))

            # Set common plot properties for linear scale
            ax_linear.set_title(f'{model_name} Performance in {region} Region (Linear Scale)',
                                fontsize=plt.rcParams['axes.titlesize'])
            ax_linear.set_ylabel('Id (µA)', fontsize=plt.rcParams['axes.labelsize'])
            ax_linear.set_xlabel('Vds (V)', fontsize=plt.rcParams['axes.labelsize'])
            ax_linear.grid(True)

            # Set common plot properties for log scale
            ax_log.set_title(f'{model_name} Performance in {region} Region (Log Scale)',
                             fontsize=plt.rcParams['axes.titlesize'])
            ax_log.set_ylabel('Id (µA)', fontsize=plt.rcParams['axes.labelsize'])
            ax_log.set_xlabel('Vds (V)', fontsize=plt.rcParams['axes.labelsize'])
            ax_log.set_yscale('log')  # Set y-axis to logarithmic scale
            ax_log.grid(True, which='both', linestyle='--', linewidth=0.5)  # Grid for both major/minor ticks

            for case_idx, case in enumerate(cases):
                label_prefix = case.get('label', f'Case {case_idx + 1}')
                w_val = case.get('W')
                l_val = case.get('L')
                vg_constant = case.get('Vg_const')
                x_range_min, x_range_max = case.get('Vds_range')
                x_range_dense = np.linspace(x_range_min, x_range_max,
                                            100)  # Generate dense Vds points for smooth curves

                # Filter the original data to get the measured subset for this specific case
                # Using np.isclose for robust float comparison
                measured_subset = full_original_data_for_plot[
                    (np.isclose(full_original_data_for_plot['w'], w_val, atol=1e-9)) &
                    (np.isclose(full_original_data_for_plot['l'], l_val, atol=1e-9)) &
                    (np.isclose(full_original_data_for_plot['vg'], vg_constant, atol=1e-2))
                    ].copy()  # Ensure working on a copy

                if measured_subset.empty:
                    print(f"  No measured data found for {label_prefix} in {region} region. Skipping this case.")
                    continue

                print(f"  Plotting case: {label_prefix}")
                # FOR DEBUGGING:
                # print(f"    Matching measured rows: {len(measured_subset)}")
                # print(f"    Example rows:\n{measured_subset[['vd', 'vg', 'vb', 'id']].head()}")

                # Assign colors based on "Best Case" or "Worst Case"
                if "Best Case" in label_prefix:
                    current_measured_color = color_best_measured
                    current_predicted_color = color_best_predicted
                elif "Worst Case" in label_prefix:
                    current_measured_color = color_worst_measured
                    current_predicted_color = color_worst_predicted
                else:  # Fallback for other cases
                    current_measured_color = 'purple'
                    current_predicted_color = 'plum'

                # Plot on both linear and log-scale axes
                self.prepare_model_input_and_predict(
                    measured_subset=measured_subset,
                    model=model,
                    ax=ax_linear,
                    x_range_dense=x_range_dense,
                    label_prefix=label_prefix,
                    measured_color=current_measured_color,
                    predicted_color=current_predicted_color
                )
                self.prepare_model_input_and_predict(
                    measured_subset=measured_subset,
                    model=model,
                    ax=ax_log,
                    x_range_dense=x_range_dense,
                    label_prefix=label_prefix,
                    measured_color=current_measured_color,
                    predicted_color=current_predicted_color
                )

            # Add legends and save figures
            ax_linear.legend(fontsize=plt.rcParams['legend.fontsize'])
            ax_log.legend(fontsize=plt.rcParams['legend.fontsize'])
            plt.tight_layout()  # Adjust layout to prevent labels from overlapping

            # Save both figures
            filename_base = f"{model_name.replace(' ', '_').lower()}_{region.lower().replace('-', '_')}_characteristics"
            linear_path = os.path.join(output_dir, f"{filename_base}_linear.png")
            log_path = os.path.join(output_dir, f"{filename_base}_log.png")

            fig_linear.savefig(linear_path)
            fig_log.savefig(log_path)
            plt.close(fig_linear)  # Close figures to free memory
            plt.close(fig_log)

            print(f"  Saved linear plot for {region} region → {linear_path}")
            print(f"  Saved log-scale plot for {region} region → {log_path}")