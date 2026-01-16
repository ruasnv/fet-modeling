# src/utils/environment.py
import os
import matplotlib.pyplot as plt
from src.core.config import settings

def setup_environment():
    """
    Sets up matplotlib environment based on global settings from main_config.yaml
    and creates the base report output directory.
    """
    report_output_dir = settings.get('paths.report_output_dir')
    
    # Apply Matplotlib style and parameters
    plt.style.use(settings.get('global_settings.matplotlib_style'))
    plt.rcParams['figure.figsize'] = settings.get('global_settings.figure_figsize')
    plt.rcParams['font.size'] = settings.get('global_settings.font_size')
    plt.rcParams['axes.labelsize'] = settings.get('global_settings.axes_labelsize')
    plt.rcParams['axes.titlesize'] = settings.get('global_settings.axes_titlesize')
    plt.rcParams['legend.fontsize'] = settings.get('global_settings.legend_fontsize')

    # Ensure the base report output directory exists
    os.makedirs(report_output_dir, exist_ok=True)
    print(f"Report output base directory created/ensured: {report_output_dir}")