# main.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
import os

# --- The Clean Imports (Your new Architecture) ---
from src.core.config import settings
from src.utils.environment import setup_environment
from src.data.processor import DataPreprocessor
from src.data.gan_handler import GANDataHandler

# Models
from src.models.baselines import SimpleNN
from src.models.transformer import PhysicsAwareTransformer
from src.models.gan import Generator, Discriminator

# Trainers & Runners
from src.training.simple_nn_trainer import NNTrainer
from src.training.gan_trainer import GANTrainer
from src.training.cv_runner import CrossValidator

# Evaluation & Utils
from src.evaluation.evaluator import NNEvaluator
from src.utils.plotter import Plotter
from src.utils.analysis import determine_characteristic_plot_cases
from src.eda.analyzer import EDAAnalyzer

def run_eda_pipeline():
    print("\n--- Running EDA Pipeline ---")
    dp = DataPreprocessor() # Loads raw data by default
    raw_df = dp.load_or_process_data(force_reprocess=True) # Just get the dataframe
    if dp.df is not None:
        analyzer = EDAAnalyzer(dp.df, title_prefix="Raw Data")
        analyzer.run_all_eda()
    else:
        print("EDA Failed: Could not load data.")

def run_gan_pipeline():
    print("\n--- Running GAN Augmentation Pipeline ---")
    
    # 1. Load and Segregate
    handler = GANDataHandler()
    segregated_data = handler.load_and_segregate_data()
    if not segregated_data: return

    scaler, _ = handler.get_scalers()
    gan_features = handler.get_features_for_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Config
    latent_dim = settings.get('gan_params.latent_dim')
    data_dim = len(gan_features)
    save_dir = Path(settings.get('paths.trained_model_dir')) / 'gans'
    
    augmented_dfs = {}
    
    # 3. Train/Generate Loop
    for region, df_region in segregated_data.items():
        print(f"\nProcessing Region: {region} ({len(df_region)} samples)")
        
        # Initialize Models
        gen = Generator(latent_dim, data_dim).to(device)
        disc = Discriminator(data_dim).to(device)
        
        trainer = GANTrainer(gen, disc, device, scaler, latent_dim, save_dir, region)
        
        # Load or Train
        if settings.get('run_flags.skip_training_if_exists') and trainer.load_models():
            print(f"  Loaded existing GAN for {region}.")
        else:
            print(f"  Training new GAN for {region}...")
            trainer.train_gan(df_region, settings.get('gan_params.num_gan_epochs'), settings.get('gan_params.gan_batch_size'))

        # Generate Data (Balance the dataset)
        # For simplicity, we double the data for minority classes, or define a target size
        target_size = 5000 # Example target
        current_size = len(df_region)
        if current_size < target_size:
            needed = target_size - current_size
            print(f"  Generating {needed} synthetic samples...")
            augmented_dfs[region] = trainer.generate_synthetic_data(needed)
        else:
            augmented_dfs[region] = pd.DataFrame(columns=df_region.columns)

    # 4. Save
    output_path = Path(settings.get('paths.aug_data_dir')) / 'augmented_data.pkl'
    handler.combine_and_save_augmented_data(augmented_dfs, output_path)

def run_training_pipeline(model_type, use_cv=False):
    print(f"\n--- Running Training Pipeline ({model_type}) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    # Determine if we use Raw or Augmented data based on Config
    # For now, let's assume we use the standard Processed data
    dp = DataPreprocessor()
    if not dp.load_or_process_data(): return

    X_cv, y_cv, X_cv_df, fold_indices = dp.get_cv_data()
    input_dim = X_cv.shape[1]
    
    # 2. Define Model
    if model_type == "simplenn":
        model_class = SimpleNN
        model_params = {"input_dim": input_dim}
    elif model_type == "transformer":
        model_class = PhysicsAwareTransformer
        model_params = {"input_dim": input_dim, "d_model": 64, "n_heads": 4}

    # 3. Cross Validation Mode
    if use_cv:
        print(f"Starting CV for {model_type}...")
        validator = CrossValidator(device, nn.MSELoss(), *dp.get_scalers(), dp.get_features_for_model())
        validator.run_cv(
            model_class, NNTrainer, X_cv, y_cv, X_cv_df, fold_indices,
            model_params,
            training_params={
                "num_epochs": settings.get('training_params.num_epochs'),
                "batch_size": settings.get('training_params.batch_size'),
                "learning_rate": settings.get('training_params.learning_rate'),
                "model_name": model_type
            },
            model_save_base_dir=settings.get('paths.trained_model_dir'),
            report_output_dir=settings.get('paths.report_output_dir')
        )
    
    # 4. Final Training Mode
    else:
        print(f"Training Final {model_type}...")
        model = model_class(**model_params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=settings.get('training_params.learning_rate'))
        save_path = Path(settings.get('paths.trained_model_dir')) / f'{model_type}_final.pth'
        
        trainer = NNTrainer(model, device, nn.MSELoss(), optimizer, save_path)
        
        # Train on EVERYTHING (CV Pool + Final Test is handled by DP, usually we train on CV pool here)
        # Or combine them. For now, let's train on X_cv
        train_losses, test_losses = trainer.train(
            torch.tensor(X_cv, dtype=torch.float32),
            torch.tensor(y_cv, dtype=torch.float32),
            torch.tensor(dp.get_final_test_data()[0], dtype=torch.float32), # Validate on held-out test
            torch.tensor(dp.get_final_test_data()[1], dtype=torch.float32),
            num_epochs=settings.get('training_params.num_epochs'),
            batch_size=settings.get('training_params.batch_size')
        )
        
        # Final Plots
        plotter = Plotter(*dp.get_scalers(), dp.get_features_for_model(), device)
        cases = determine_characteristic_plot_cases(model, dp.get_filtered_original_data(), *dp.get_scalers(), dp.get_features_for_model(), device)
        if cases:
            plotter.id_vds_characteristics(model, dp.get_filtered_original_data(), cases, model_name=model_type)

def main():
    parser = argparse.ArgumentParser(description="FET Modeling Pipeline")
    parser.add_argument("--mode", type=str, choices=["train", "cv", "eda", "gan"], required=True)
    parser.add_argument("--model", type=str, choices=["simplenn", "transformer"], default="simplenn")
    args = parser.parse_args()

    setup_environment()

    if args.mode == "eda":
        run_eda_pipeline()
    elif args.mode == "gan":
        run_gan_pipeline()
    elif args.mode == "cv":
        run_training_pipeline(args.model, use_cv=True)
    elif args.mode == "train":
        run_training_pipeline(args.model, use_cv=False)

if __name__ == "__main__":
    main()