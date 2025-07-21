# src/helper.py
import torch
import pandas as pd
import numpy as np

def debug_single_prediction(model, scaler_X, scaler_y, features_for_model, device):
    """
    Debugging function to test a single input through the model and print outputs.
    """
    print("\n--- DEBUGGING SINGLE MODEL PREDICTION ---")

    sample_input_data = {
        'vg': -0.15,
        'vd': 1.0,
        'vb': 0.0,
        'w': 10.0e-6,
        'l': 3.0e-6,
        'wOverL': 10.0e-6 / 3.0e-6,
        'vgs': -0.15,
        'vds': 1.0,
        'temp': 27.0
    }

    input_df = pd.DataFrame([sample_input_data])
    input_for_scaling = input_df[features_for_model]

    print(f"Original Input Features:\n{input_for_scaling.to_string(index=False)}")

    scaled_input = scaler_X.transform(input_for_scaling)
    print(f"\nScaled Input (first 5 values): {scaled_input[0, :5]}...")

    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        output_scaled_log = model(input_tensor).cpu().numpy()

    print(f"\nModel Output (scaled log_Id): {output_scaled_log[0, 0]:.6f}")

    predicted_id_ampere = np.power(10, scaler_y.inverse_transform(output_scaled_log))[0, 0]
    predicted_id_microampere = predicted_id_ampere * 1e6

    print(f"Predicted Id (Amperes): {predicted_id_ampere:.4e}")
    print(f"Predicted Id (Microamperes): {predicted_id_microampere:.4f} µA")

def suggest_best_worst_cases(df, region_name):
    df_region = df[df['operating_region'] == region_name]

    # Group by W, L, Vg triplets
    group_cols = ['w', 'l', 'vg']
    grouped = df_region.groupby(group_cols)['id'].mean().reset_index()

    # Find the max/min average Id combinations (in Amperes)
    best_case = grouped.loc[grouped['id'].idxmax()]
    worst_case = grouped.loc[grouped['id'].idxmin()]

    def format_case(case):
        return {
            'W': case['w'],
            'L': case['l'],
            'Vg_const': case['vg'],
            'Vds_range': [0, 4],
            'label': f"W={case['w'] * 1e6:.1f}µm, L={case['l'] * 1e6:.1f}µm, Vg={case['vg']:.2f}V"
        }

    return format_case(best_case), format_case(worst_case)


def determine_best_worst_ranges(full_filtered_original_df):
    cutoff_best, cutoff_worst = suggest_best_worst_cases(full_filtered_original_df, "Cut-off")
    linear_best, linear_worst = suggest_best_worst_cases(full_filtered_original_df, "Linear")
    saturation_best, saturation_worst = suggest_best_worst_cases(full_filtered_original_df, "Saturation")

    test = {
        'Cut-off': {'id_vs_vds': [cutoff_best, cutoff_worst]},
        'Linear': {'id_vs_vds': [linear_best, linear_worst]},
        'Saturation': {'id_vs_vds': [saturation_best, saturation_worst]}
    }

    return test

def calculate_vth(vsb, vth0=0.5, gamma=0.5, phi_f=0.35):
    return vth0 + gamma * (np.sqrt(np.abs(vsb + 2 * phi_f)) - np.sqrt(2 * phi_f))


def classify_region(row):
    Vgs_val, Vds_val, Vth_val = row['vgs'], row['vds'], row['vth']
    if Vgs_val < Vth_val:
        return 'Cut-off'
    elif Vds_val < (Vgs_val - Vth_val):
        return 'Linear'
    else:
        return 'Saturation'

