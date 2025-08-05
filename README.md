MOSFET Id Modeling Project
This project provides a comprehensive framework for modeling the drain current (I_D) of MOSFETs using a deep learning approach. It encompasses data loading, preprocessing, exploratory data analysis (EDA), neural network training with cross-validation, model evaluation, and characteristic curve plotting.

Key Features
Modular Design: Organized into distinct Python modules for data processing, modeling, training, evaluation, cross-validation, and utilities.

Configurable Workflow: Uses YAML configuration files (configs/) to manage parameters for data filtering, feature engineering, model architecture, training, and plotting.

Robust Data Preprocessing:

Handles raw CSV data loading and column name cleaning.

Applies configurable filters (e.g., I_D0, V_DS0, specific temperature).

Performs feature engineering, including W/L ratio, V_GS, V_DS, V_BS, and log-transformation of I_D.

Dynamically calculates threshold voltage (V_TH) using the body effect formula.

Classifies operating regions (Cut-off, Linear, Saturation) for stratified data splitting.

Scales features and target using StandardScaler.

Exploratory Data Analysis (EDA):

Provides insights into data distribution, missing values, and basic statistics.

Visualizes temperature distribution and device size variations.

Analyzes operating region distributions.

Generates feature correlation heatmaps.

Neural Network Modeling:

Implements a SimpleNN (feedforward neural network) for I_D prediction.

Uses PyTorch for model definition and training.

Flexible Training & Evaluation:

NNTrainer class handles the training loop, loss calculation, and model saving/loading.

NNEvaluator class calculates key regression metrics (R 
2
 , MAE, RMSE on both scaled and original data, and MAPE on original data).

K-Fold Cross-Validation:

CrossValidator orchestrates stratified K-fold cross-validation to assess model generalization.

Saves detailed and summary metrics for each fold.

Characteristic Curve Plotting:

Plotter generates I_D-V_DS and I_D-V_GS characteristic curves.

Compares model predictions against measured data on both linear and logarithmic scales.

Supports plotting for specific device sizes and operating conditions defined in the configuration.

Getting Started
Follow these steps to set up and run the project on your local machine.

Prerequisites
Python 3.8+: Ensure you have a compatible Python version installed.

pip: Python package installer (usually comes with Python).

Installation
Clone the repository:

git clone <repository_url>
cd fetModeling

Create a virtual environment (recommended):

python -m venv .venv

Activate the virtual environment:

On Windows:

.venv\Scripts\activate

On macOS/Linux:

source .venv/bin/activate

Install project dependencies:

pip install -r requirements.txt

(If requirements.txt is not provided, you'll need to create it with the following packages: pandas, numpy, scikit-learn, matplotlib, seaborn, torch, pyyaml, joblib)

Project Structure
fetModeling/
├── configs/
│   ├── data_config.yaml          # Data processing and splitting configuration
│   ├── main_config.yaml          # Main project settings (paths, global flags)
│   └── simple_nn_config.yaml     # SimpleNN model and training parameters
├── data/
│   └── nmoshv.csv                # Raw MOSFET measurement data
├── models/
│   └── trained/                  # Directory for saving trained models (final & CV folds)
├── results/
│   ├── eda/                      # Output directory for EDA plots and reports
│   ├── plots/                    # Output directory for general plots
│   └── reports/                  # Output directory for training reports (CV summaries, etc.)
├── scripts/
│   ├── run_data_processing.py    # Script to execute data preprocessing
│   ├── run_eda.py                # Script to execute exploratory data analysis
│   └── run_training_simple_nn.py # Script to train and evaluate the SimpleNN model
├── src/
│   ├── config.py                 # Handles loading and merging configuration files
│   ├── data_processing/
│   │   ├── data_loader.py        # Loads raw data and cleans column names
│   │   └── preprocessor.py       # Filters, engineers features, and splits data
│   ├── eda/
│   │   └── analyzer.py           # Performs EDA and generates plots
│   ├── models/
│   │   └── simple_nn.py          # Defines the SimpleNN architecture
│   ├── training/
│   │   └── simple_nn_trainer.py  # Manages model training, saving, and loss plotting
│   ├── evaluation/
│   │   └── evaluator.py          # Calculates and reports model performance metrics
│   ├── cross_validation/
│   │   └── cv_runner.py          # Orchestrates K-fold cross-validation
│   └── utils/
│       ├── helpers.py            # Utility functions (env setup, Vth, region classification)
│       └── plotter.py            # Generates characteristic plots
└── README.md                     # This documentation file
└── requirements.txt              # List of Python dependencies

Core Components & Scripts
Configuration Management (src/config.py & configs/)
The project uses a centralized configuration system. Parameters are defined in YAML files within the configs/ directory. src/config.py is responsible for loading and merging these configurations into a single, accessible settings object, which can then be used throughout the project.

configs/data_config.yaml: Defines parameters related to data filtering, feature engineering, normalization, V_TH calculation, and data splitting.

configs/main_config.yaml: Contains global settings such as output paths, Matplotlib styling, and flags to control script execution (e.g., run_cross_validation).

configs/simple_nn_config.yaml: Specifies the SimpleNN model's input dimension, training parameters (epochs, batch size, learning rate, criterion, optimizer), and plotting cases.

Utility Functions (src/utils/helpers.py)
This module provides essential helper functions:

setup_environment(): Configures global settings like Matplotlib styles and ensures output directories exist.

calculate_vth(): Computes the MOSFET threshold voltage (V_TH) using the body effect formula, taking into account V_SB, V_TH0, 
gamma, and 
phi_F.

classify_region(): Classifies the MOSFET operating region (Cut-off, Linear, Saturation) based on V_GS, V_DS, and V_TH.

Data Processing (src/data_processing/)
data_loader.py:

Loads the raw MOSFET measurement data from a CSV file (data/nmoshv.csv).

Performs initial cleaning of column names (e.g., removing leading/trailing whitespace) to ensure consistent access.

preprocessor.py:

Applies configurable filters to the raw data (e.g., id > 0, vds > 0, specific temp).

Engineers new features: vgs, vds, vbs, vsb, wOverL (Width-to-Length ratio), and log_Id (log10 of drain current).

Calculates the dynamic vth using the body effect formula and classifies the operating_region.

Splits the data into cross-validation (CV) pool and final test sets, using stratified sampling based on operating_region to ensure balanced representation.

Scales numerical features and the target (log_Id) using StandardScaler for neural network compatibility.

Saves the processed data and scalers to data/processed/ for reusability.

Exploratory Data Analysis (EDA) (src/eda/analyzer.py)
The EDAAnalyzer class performs a comprehensive analysis of the raw dataset. It's designed to provide insights into the data's characteristics before heavy filtering for model training.

Data Overview: Prints shape, memory usage, and available columns.

Missing Values: Identifies and reports any missing data.

Basic Statistics: Provides descriptive statistics (.describe()) and a peek at the first few rows.

Temperature Distribution: Visualizes the counts of data points at different temperatures across the entire dataset.

Device Size Analysis: Reports unique device sizes (W, L) and visualizes the number of unique temperature measurements and total data points per device size using heatmaps.

Operating Region Distribution: Classifies and plots the distribution of operating regions for a specific temperature (as defined in data_config.yaml).

Feature Correlation Matrix: Generates a heatmap showing the correlation between selected features (e.g., vg, vd, wOverL, log_Id).

Neural Network Model (src/models/simple_nn.py)
SimpleNN: A basic feedforward neural network built with PyTorch. It consists of multiple linear layers with GELU activation functions and a final output layer. The input_dim is the only configurable parameter for its __init__ method, with hidden layer dimensions being hardcoded internally.

Training (src/training/simple_nn_trainer.py)
NNTrainer: Manages the training lifecycle of a PyTorch model.

Takes an instantiated model, device, loss function (criterion), and optimizer as inputs.

Implements the training loop, including batch processing and epoch iteration.

Calculates and tracks training and validation losses.

Saves the best-performing model (based on validation loss) to disk.

Provides a plot_losses method to visualize training and testing loss curves.

Evaluation (src/evaluation/evaluator.py)
NNEvaluator: Assesses the performance of a trained model.

Calculates standard regression metrics:

R 
2
  Score: Measures the proportion of variance in the dependent variable that can be predicted from the independent variables.

Mean Absolute Error (MAE): Average of the absolute differences between predictions and actual values.

Root Mean Squared Error (RMSE): Square root of the average of the squared differences between predictions and actual values.

Calculates these metrics on both the scaled (log-transformed) data (what the model directly optimizes) and the original (physical) scale of the drain current (I_D).

Includes Mean Absolute Percentage Error (MAPE) on the original I_D scale, which is crucial for understanding percentage errors relative to the true value, especially important for small currents. It handles potential division-by-zero issues by clipping very small true values.

Cross-Validation (src/cross_validation/cv_runner.py)
CrossValidator: Orchestrates the K-fold cross-validation process.

Splits the CV pool data into training and validation folds based on the cv_fold_indices generated by DataPreprocessor.

Trains a new instance of the model for each fold using NNTrainer.

Evaluates the model's performance on the validation set of each fold using NNEvaluator.

Collects and aggregates metrics across all folds to provide mean and standard deviation for robust performance assessment.

Saves detailed metrics for each fold and a summary report to results/reports/.

Saves individual fold models to models/trained/k_fold_models/.

Plotting Utilities (src/utils/plotter.py)
Plotter: A dedicated class for generating characteristic curves.

prepare_model_input_and_predict(): A helper method that generates synthetic input data for specific device conditions, runs the trained model to get predictions, inverse transforms them back to the original I_D scale, and prepares them for plotting against measured data.

id_vds_characteristics(): Generates I_D-V_DS plots (or I_D-V_GS if configured) for specific device sizes and operating conditions. It compares the model's predictions against interpolated measured data on both linear and logarithmic scales, providing clear visual insights into model accuracy. Plots are saved to results/plots/characteristic_plots/.

How to Run the Project (Workflow)
Follow these steps in order to run the full pipeline:

Activate your virtual environment.

Windows: .\.venv\Scripts\activate

macOS/Linux: source ./.venv/bin/activate

Run Data Preprocessing:
This step loads raw data, applies filters, engineers features, calculates V_TH and operating regions, scales data, and splits it into CV and final test sets. Processed data is saved for later use.

python scripts/run_data_processing.py

Run Exploratory Data Analysis (EDA):
This step generates various plots and a detailed report (.txt file) in results/eda/ that summarize the characteristics of your raw dataset.

python scripts/run_eda.py

Run Model Training and Evaluation:
This is the main script for training your neural network. It will:

Load the processed data.

Optionally run K-fold cross-validation (controlled by run_flags.run_cross_validation in main_config.yaml).

Train a final SimpleNN model on the entire CV pool.

Evaluate the final model on the held-out test set.

Generate characteristic plots comparing predictions to measured data (controlled by run_flags.skip_plots_if_exists in main_config.yaml).

All training logs, evaluation reports, and plots will be saved in the results/ and models/trained/ directories.

python scripts/run_training_simple_nn.py

Key Python Packages
This project heavily relies on the following Python libraries:

pandas: Essential for data manipulation and analysis, especially with DataFrames.

numpy: Provides powerful numerical computing capabilities, fundamental for array operations.

scikit-learn: Offers machine learning tools, particularly StandardScaler for data normalization and various sklearn.metrics for evaluation.

matplotlib: A foundational plotting library used for generating all visualizations.

seaborn: Built on Matplotlib, it provides a high-level interface for drawing attractive statistical graphics, used for heatmaps and bar plots.

torch (PyTorch): The deep learning framework used for building, training, and evaluating the neural network models.

PyYAML: For parsing and managing configuration files in YAML format.

joblib: For efficient saving and loading of Python objects, such as StandardScaler instances and processed DataFrames.

scipy.interpolate.interp1d: Used in the Plotter for interpolating measured data to create smooth characteristic curves.