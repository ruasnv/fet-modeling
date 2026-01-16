# src/physics/equations.py
import numpy as np
import pandas as pd

# TODO: This method of calculating the Vth was added later using SPICE Level 1.
# Suggest checking this theoretically with a domain expert.
def calculate_dynamic_vth(vsb, vth0=0.7, gamma=0.4, phi_f=0.4):
    """
    Calculates the threshold voltage (Vth) using the body effect formula.
    Ref: Tsividis, Operation and Modeling of the MOS Transistor.
    """
    # Common formula for body effect: Vth = Vth0 + gamma * (sqrt(2*phi_f + V_SB) - sqrt(2*phi_f))
    # `2*phi_f + vsb` could become negative. Clipping to a small positive value prevents NaNs.
    sqrt_term_arg = np.maximum(2 * phi_f + vsb, 1e-9)
    
    return vth0 + gamma * (np.sqrt(sqrt_term_arg) - np.sqrt(2 * phi_f))

def classify_region(row, vth_approx_val):
    """
    Classifies the operating region of a MOSFET based on Vgs, Vds, and Vth.
    Returns: 'Cut-off', 'Linear', or 'Saturation'.
    """
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