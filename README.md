# MOSFET Id Modeling Project

This project provides a comprehensive framework for modeling the drain current (**I<sub>D</sub>**) of MOSFETs using a deep learning approach.  
It encompasses data loading, preprocessing, exploratory data analysis (EDA), neural network training with cross-validation, model evaluation, and characteristic curve plotting.

---

## **Key Features**

### **Modular Design**
Organized into distinct Python modules for:
- Data processing
- Modeling
- Training
- Evaluation
- Cross-validation
- Utilities

### **Configurable Workflow**
- Uses YAML configuration files (`configs/`) to manage parameters for data filtering, feature engineering, model architecture, training, and plotting.

### **Robust Data Preprocessing**
- Handles raw CSV data loading and column name cleaning.
- Applies configurable filters (e.g., I_D0, V_DS0, specific temperature).
- Performs feature engineering, including W/L ratio, V_GS, V_DS, V_BS, and log-transformation of I_D.
- Dynamically calculates threshold voltage (V_TH) using the body effect formula.
- Classifies operating regions (Cut-off, Linear, Saturation) for stratified data splitting.
- Scales features and target using `StandardScaler`.

### **Exploratory Data Analysis (EDA)**
- Provides insights into data distribution, missing values, and basic statistics.
- Visualizes temperature distribution and device size variations.
- Analyzes operating region distributions.
- Generates feature correlation heatmaps.

### **Neural Network Modeling**
- Implements a **SimpleNN** (feedforward neural network) for I_D prediction.
- Uses **PyTorch** for model definition and training.

### **Flexible Training & Evaluation**
- `NNTrainer` class handles the training loop, loss calculation, and model saving/loading.
- `NNEvaluator` class calculates key regression metrics (R², MAE, RMSE on both scaled and original data, and MAPE on original data).

### **K-Fold Cross-Validation**
- `CrossValidator` orchestrates stratified K-fold cross-validation to assess model generalization.
- Saves detailed and summary metrics for each fold.

### **Characteristic Curve Plotting**
- `Plotter` generates I_D-V_DS and I_D-V_GS characteristic curves.
- Compares model predictions against measured data on both linear and logarithmic scales.
- Supports plotting for specific device sizes and operating conditions defined in the configuration.

---

## **Getting Started**

Follow these steps to set up and run the project locally.

### **Prerequisites**
- **Python 3.8+**: Ensure you have a compatible Python version installed.
- **pip**: Python package installer (usually included with Python).

### **Installation**
Clone the repository:
```
git clone <repository_url>
cd fetModeling

```
Create a virtual environment (recommended):
```
python -m venv .venv
```

Project Structure
```
fetModeling/
├── configs/
│   ├── data_config.yaml         # Data processing and splitting configuration
│   ├── main_config.yaml         # Main project settings (paths, global flags)
│   └── simple_nn_config.yaml    # SimpleNN model and training parameters
├── data/
│   └── nmoshv.csv               # Raw MOSFET measurement data
├── models/
│   └── trained/                 # Directory for trained models (final & CV folds)
├── results/
│   ├── eda/                     # Output directory for EDA plots/reports
│   ├── plots/                   # Output directory for plots
│   └── reports/                 # Output directory for training/CV summaries
├── scripts/
│   ├── run_data_processing.py   # Execute data preprocessing
│   ├── run_eda.py               # Execute EDA
│   └── run_training_simple_nn.py# Train and evaluate the SimpleNN model
├── src/
│   ├── config.py                # Load and merge configuration files
│   ├── data_processing/
│   │   ├── data_loader.py       # Load raw data, clean column names
│   │   └── preprocessor.py      # Filter, engineer features, split data
│   ├── eda/
│   │   └── analyzer.py          # Perform EDA and generate plots
│   ├── models/
│   │   └── simple_nn.py         # SimpleNN architecture
│   ├── training/
│   │   └── simple_nn_trainer.py # Manage training, saving, loss plotting
│   ├── evaluation/
│   │   └── evaluator.py         # Model performance metrics
│   ├── cross_validation/
│   │   └── cv_runner.py         # Run K-fold cross-validation
│   └── utils/
│       ├── helpers.py           # Utility functions (env setup, Vth, region classification)
│       └── plotter.py           # Generate characteristic plots
└── README.md
└── requirements.txt
```

## **Core Components & Scripts**

### **Configuration Management**
Centralized configuration system in configs/

Loaded via src/config.py

## **Key files**:

- main_config.yaml: Global settings, paths, flags.

## Utility Functions (src/utils/helpers.py)
setup_environment(): Configures Matplotlib styles, ensures directories exist.

calculate_vth(): Computes V_TH using body effect formula.

classify_region(): Determines Cut-off, Linear, or Saturation region.

## Data Processing
data_loader.py: Loads raw CSV, cleans column names.

preprocessor.py:

Filters raw data (id > 0, vds > 0, etc.).

Engineers features (vgs, vds, vbs, vsb, wOverL, log_Id).

Calculates dynamic V_TH and operating_region.

Splits into CV pool & test sets (stratified sampling).

Scales using StandardScaler.

Saves processed data & scalers.

## EDA
Analyzes dataset statistics, missing values, and distributions.

Temperature distribution, device size heatmaps.

Operating region distribution.

Feature correlation heatmaps.

## Neural Network Model
SimpleNN: Feedforward PyTorch model with GELU activation.

## Training
NNTrainer: Handles training loop, validation, saving best model, plotting losses.

## Evaluation
NNEvaluator: Computes:

R², MAE, RMSE (scaled & original)

MAPE (original scale, with small value handling)

## Cross-Validation
CrossValidator: Runs K-fold CV, trains/evaluates per fold.

Saves fold metrics, summary, and trained models.

## Plotting
Plotter:

Generates synthetic inputs, predicts with model.

Creates I_D-V_DS and I_D-V_GS plots (linear/log scale).

Compares predictions vs measured data.

##How to Run the Project (Workflow)

Activate the virtual environment:
Windows
```
.venv\Scripts\activate
```
macOS/Linux
```
source .venv/bin/activate
```

Install project dependencies:
```
pip install -r requirements.txt
```
Run the script
```
python clean_rerun.py
```

**Key Python Packages**

pandas
numpy
scikit-learn
matplotlib
seaborn
torch
pyyaml
joblib
scipy.interpolate.interp1d