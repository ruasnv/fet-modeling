# config.py

# --- Paths ---
RAW_DATA_PATH = 'data/nmoshv.csv'
PROCESSED_DATA_DIR = 'data/processed_data'
TRAINED_MODEL_DIR = 'trained_models'
REPORT_OUTPUT_DIR = 'reports/models'

# --- Data Filtering ---
TEMP_FILTER = 27.0
CV_FINAL_TEST_SPLIT_RATIO = 0.1
NUM_FOLDS = 10
STRATIFY_COLUMN = 'operating_region'

# --- Model Hyperparameters ---
NN_INPUT_DIM = 4
NN_NUM_EPOCHS = 100
NN_BATCH_SIZE = 64
NN_LEARNING_RATE = 0.001

# --- Global Flags ---
SKIP_TRAINING_IF_EXISTS = True
SKIP_PLOTS_IF_EXISTS = True

# --- Plotting Configurations ---
cases_config_for_best_worst_plots = {
    'Cut-off': [
        {'label': 'Best Case (W=0.3µm, L=0.3µm, Vg=0.45V)', 'W': 1e-05, 'L': 3e-07, 'Vg_const': 0.65,
         'Vds_range': [0, 4]},
        {'label': 'Worst Case (W=10µm, L=3µm, Vg=-0.15V)', 'W': 6e-07, 'L': 1e-05, 'Vg_const': -0.05,
         'Vds_range': [0, 4]}
    ],
    'Linear': [
        {'label': 'Best Case (W=0.6µm, L=1.2µm, Vg=2.9V)', 'W': 1e-05, 'L': 3e-07, 'Vg_const': 3.6,
         'Vds_range': [0, 3]},
        {'label': 'Worst Case (W=10µm, L=1µm, Vg=2.0V)', 'W': 3e-07, 'L': 1e-05, 'Vg_const': 1.148,
         'Vds_range': [0, 3]}
    ],
    'Saturation': [
        {'label': 'Best Case (W=10µm, L=0.3µm, Vg=2.25V)', 'W': 1e-05, 'L': 3e-07, 'Vg_const': 3.6,
         'Vds_range': [0, 4]},
        {'label': 'Worst Case (W=10µm, L=10µm, Vg=0.9V)', 'W': 3e-07, 'L': 1e-05, 'Vg_const': 1.148,
         'Vds_range': [0, 4]}
    ]
}
