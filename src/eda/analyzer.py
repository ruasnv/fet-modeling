# src/eda/analyzer.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from src.core.config import settings
from src.physics.equations import classify_region, calculate_dynamic_vth

class EDAAnalyzer:
    def __init__(self, df, output_subdir="eda", title_prefix=""):
        """
        Universal EDA Analyzer for both Raw and Augmented Data.
        
        Args:
            df: DataFrame to analyze.
            output_subdir: Subdirectory in 'results/' to save plots (e.g., 'eda' or 'eda_augmented').
            title_prefix: String to add to plot titles (e.g. "Augmented Data").
        """
        self.df = df.copy()
        self.output_dir = Path(settings.get('paths.report_output_dir')) / output_subdir
        self.title_prefix = title_prefix
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        if settings.get("global_settings.ignore_warnings", True):
            warnings.filterwarnings('ignore')
            
        self._prepare_derived_features()

    def _prepare_derived_features(self):
        """Calculates Vgs, Vds, W/L, etc. if missing."""
        # Voltages
        if 'vs' in self.df.columns:
            for term in ['vg', 'vd', 'vb']:
                if term in self.df.columns and f'{term}s' not in self.df.columns:
                    self.df[f'{term}s'] = self.df[term] - self.df['vs']
        
        # Geometry
        if 'w' in self.df.columns and 'l' in self.df.columns and 'wOverL' not in self.df.columns:
            self.df['wOverL'] = self.df['w'] / self.df['l']
            
        # Log Current
        if 'id' in self.df.columns and 'log_Id' not in self.df.columns:
            min_current = 1e-12
            self.df['id_clipped'] = np.clip(self.df['id'], a_min=min_current, a_max=None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.df['log_Id'] = np.log10(self.df['id_clipped'])

        # Physics (Vth)
        if 'vth' not in self.df.columns:
            # Ensure Vsb exists
            if 'vsb' not in self.df.columns and 'vs' in self.df.columns:
                 self.df['vsb'] = self.df['vs'] - self.df['vb']
            
            # Calculate Dynamic Vth
            if 'vsb' in self.df.columns:
                vth_params = settings.get('vth_params')
                self.df['vth'] = self.df['vsb'].apply(
                    lambda x: calculate_dynamic_vth(x, 
                        vth0=vth_params['vth0'], 
                        gamma=vth_params['gamma'], 
                        phi_f=vth_params['phi_f'])
                )

    def run_all_eda(self):
        """Runs the standard suite of analysis."""
        print(f"\n--- Running EDA ({self.title_prefix}) ---")
        self._plot_device_size_distribution()
        self._analyze_operating_regions()
        self._analyze_correlations()
        print(f"Plots saved to: {self.output_dir}")

    def _plot_device_size_distribution(self):
        """Scatter plot of W vs L."""
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.df, x='l', y='w', alpha=0.6)
        plt.title(f'{self.title_prefix} Device Sizes (W vs L)')
        plt.xlabel('Length (m)')
        plt.ylabel('Width (m)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(self.output_dir / 'device_sizes_scatter.png')
        plt.close()

    def _analyze_operating_regions(self):
        """Recalculates regions to ensure consistency."""
        # Re-classify based on current Vth (Physics Check)
        self.df['calc_region'] = self.df.apply(
            lambda row: classify_region(row, vth_approx_val=row['vth']), axis=1
        )
        
        region_counts = self.df['calc_region'].value_counts().sort_index()
        print(f"\nOperating Regions ({self.title_prefix}):")
        print(region_counts)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=region_counts.index, y=region_counts.values, palette='viridis')
        plt.title(f'{self.title_prefix} Operating Region Distribution')
        plt.savefig(self.output_dir / 'operating_regions.png')
        plt.close()

    def _analyze_correlations(self):
        """Heatmap of feature correlations."""
        cols = ['vgs', 'vds', 'vbs', 'w', 'l', 'wOverL', 'log_Id']
        existing = [c for c in cols if c in self.df.columns]
        
        if len(existing) > 1:
            corr = self.df[existing].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f'{self.title_prefix} Correlation Matrix')
            plt.savefig(self.output_dir / 'correlation_matrix.png')
            plt.close()