# MOSFET I_D Modeling Project
This project provides a robust framework for modeling the drain current (I_D) of MOSFETs using a data-driven approach. It addresses the challenge of limited real-world measurement data by employing a specialized Physically-Aware Generative Adversarial Network (GAN) for data augmentation. The final compact model is trained on a comprehensive, augmented dataset to ensure high accuracy and generalization.

## **Key Features**

### Data Augmentation with Physically-Aware GAN
Our core innovation is a GAN designed to generate synthetic data that adheres to fundamental physical laws. The Generator network is trained to produce only the core physical parameters that cannot be derived from others.

Device dimensions: Width (W) and Length (L)

Operating voltages: Gate-Source (V_gs) and Drain-Source (V_ds)

Drain current: (I_D)

A key architectural feature is the use of the Softplus activation function on the final output layer for parameters that must be positive, such as W, L, and I_D. This guarantees that all generated device sizes and currents are physically plausible. Other parameters, such as the W/L ratio and operating region, are derived from the generated core values, ensuring mathematical consistency.

### Modular Design
The project is organized into distinct Python modules for data processing, modeling, training, evaluation, and utilities, providing a clean and maintainable structure.

### Configurable Workflow
The entire workflow is managed through YAML configuration files (configs/). This allows for easy adjustment of parameters for data filtering, feature engineering, model architecture, training, and plotting without changing the code.

### Data Preprocessing
The pipeline handles raw CSV data, cleans column names, and performs advanced feature engineering, including the calculation of W/L ratio, V_gs, V_ds, V_bs, and log-transformation of I_D. It also dynamically calculates the threshold voltage (V_th) and classifies operating regions for stratified data splitting.

### Exploratory Data Analysis (EDA)
The EDA module provides in-depth insights into the dataset's characteristics, including missing values, basic statistics, temperature distribution, device size variations, and feature correlation heatmaps.

### Neural Network Modeling
The core of the compact model is a SimpleNN (feedforward neural network) implemented in PyTorch. The framework is flexible and can be adapted to other network architectures.

### Flexible Training & Evaluation
The NNTrainer and NNEvaluator classes handle the training loop, loss calculation, and performance metrics (R², MAE, RMSE, and MAPE) to ensure the model's accuracy.

### K-Fold Cross-Validation
A CrossValidator orchestrates stratified K-fold cross-validation to assess the model's generalization capabilities across different data splits.

### Characteristic Curve Plotting
A Plotter class generates I_D-V_DS and I_D-V_GS characteristic curves, comparing model predictions against measured data on both linear and logarithmic scales for visual verification.

## Getting Started
### Prerequisites
Python 3.8+

### Installation
Clone the repository:
```
git clone https://github.com/ruasnv/mosfet-id-modeling.git
cd mosfet-id-modeling
```
Create and activate a virtual environment (recommended):
```
python -m venv .venv
```
```
# On Windows
.venv\Scripts\activate
```
```
# On macOS/Linux
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```
Running the Project
The entire pipeline, from data processing and GAN augmentation to final model training and evaluation, is orchestrated by a single script.
```
python src/main.py
```
This will generate synthetic data, train the final model, and save all results to the models/ and results/ directories.