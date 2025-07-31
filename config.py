# config.py

# --- Paths ---
RAW_DATA_PATH = 'data/nmoshv.csv'
PROCESSED_DATA_DIR = 'data/processed'
TRAINED_MODEL_DIR = 'models'
REPORT_OUTPUT_DIR = 'results/models'

# --- Data Filtering ---
TEMP_FILTER = 27.0
CV_FINAL_TEST_SPLIT_RATIO = 0.1
NUM_FOLDS = 10
STRATIFY_COLUMN = 'operating_region'

# --- Model Hyperparameters ---
NN_INPUT_DIM = 5
NN_NUM_EPOCHS = 100
NN_BATCH_SIZE = 64
NN_LEARNING_RATE = 0.001

# --- Global Flags ---
SKIP_TRAINING_IF_EXISTS = True
SKIP_PLOTS_IF_EXISTS = True

# --- Plotting Configurations ---
cases_config_for_best_worst_plots = {
    'Cut-off': [
        {'label': 'Best Case', 'W': 1e-05, 'L': 3e-07, 'Vg_const': 0.65,
         'Vds_range': [0, 4]},
        {'label': 'Worst Case', 'W': 6e-07, 'L': 1e-05, 'Vg_const': -0.05,
         'Vds_range': [0, 4]}
    ],
    'Linear': [
        {'label': 'Best Case', 'W': 1e-05, 'L': 3e-07, 'Vg_const': 3.6,
         'Vds_range': [0, 3]},
        {'label': 'Worst Case', 'W': 3e-07, 'L': 1e-05, 'Vg_const': 1.148,
         'Vds_range': [0, 3]}
    ],
    'Saturation': [
        {'label': 'Best Case', 'W': 1e-05, 'L': 3e-07, 'Vg_const': 3.6,
         'Vds_range': [0, 4]},
        {'label': 'Worst Case', 'W': 3e-07, 'L': 1e-05, 'Vg_const': 1.148,
         'Vds_range': [0, 4]}
    ]
}


