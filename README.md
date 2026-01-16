FET Modeling with Deep Learning & GAN-Based Augmentation
========================================================

This repository implements a robust pipeline for modeling the DC characteristics of **NMOS-HV transistors** using Deep Neural Networks (DNNs). It addresses the challenge of data scarcity in semiconductor device modeling by employing **Generative Adversarial Networks (GANs)** to synthesize physically realistic measurement data.

**Key Features:**

-   **Physics-Informed Feature Engineering:** Incorporates domain-specific features like $W/L$ ratio and dynamic Threshold Voltage ($V_{th}$) calculation using the SPICE Level 1 Body Effect formula.

-   **GAN Data Augmentation:** A specialized GAN architecture designed to generate tabular device measurement data, balancing the dataset across Cut-off, Linear, and Saturation regions.

-   **Modular Pipeline:** A production-ready Python package structure separating data processing, training, and evaluation.

-   **Reproducible Results:** Fully configurable experiments via YAML files and fixed random seeds.

* * * * *

Project Structure
--------------------

Plaintext

```
fet_modeling/
├── configs/                 # Configuration files
│   └── baseline.yaml        # Main config for Training and GANs
├── data/                    # Data storage (Raw and Processed)
├── tests/                   # Tests
│   └── check_data_loading.py
├── tools/                   # Tools
│   └── convert_pkl_to_csv.py
├── src/                     # Source code
│   ├── core/                # Config management
│   ├── data/                # Data loading & preprocessing pipelines
│   ├── eda/                 # Exploratory Data Analysis tools
│   ├── models/              # PyTorch definitions (SimpleNN, GAN)
│   ├── physics/             # Semiconductor equations (Vth, Classification)
│   ├── training/            # Training loops & Cross-Validation
│   └── utils/               # Plotting & Analysis helpers
├── main.py                  # Single entry point for the pipeline
└── requirements.txt         # Dependencies

```

* * * * *

Getting Started
------------------

### 1\. Installation

Clone the repository and set up a Python virtual environment (Python 3.8+ recommended).

Bash

```
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### 2\. Configuration

All hyperparameters (learning rate, batch size, physics constants) are defined in `configs/baseline.yaml`. You do not need to change the code to experiment with different settings.

* * * * *

How to Run the Pipeline
--------------------------

The project uses a single entry point `main.py`. You control the workflow using the `--mode` argument.

### Step 1: Exploratory Data Analysis (EDA)

Generates distribution plots, correlation matrices, and physics consistency checks for the raw data.

Bash

```
python main.py --mode eda

```

*Outputs are saved to:* `results/eda/`

### Step 2: Cross-Validation (Benchmarking)

Runs K-Fold Cross-Validation on the `SimpleNN` model to establish a performance baseline. This uses the *Physics-Informed features* ($W/L, V_{th}$) to achieve high accuracy.

Bash

```
python main.py --mode cv --model simplenn

```

*Outputs:* Summary metrics (RMSE, MAPE) saved to `results/`.

### Step 3: GAN Augmentation

Trains a Generator/Discriminator pair for specific operating regions (e.g., Cut-off) to generate synthetic data.

Bash

```
python main.py --mode gan

```

*Effect:* Trains GANs, generates synthetic samples, and saves a combined dataset to `data/processed/`.

### Step 4: Final Training

Trains the final `SimpleNN` regressor on the full (or augmented) dataset and saves the model for deployment.

Bash

```
python main.py --mode train --model simplenn

```

*Outputs:* Trained model saved to `models/final_model.pth`.

* * * * *

Methodology & Physics
------------------------

### Feature Engineering

Instead of feeding raw dimensions, we pre-process inputs to align with device physics:

1.  **Geometric Ratio:** $W/L$ is explicitly calculated as it linearly correlates with current in the saturation region.

2.  Dynamic Threshold Voltage ($V_{th}$):

    $$V_{th} = V_{th0} + \gamma (\sqrt{2\phi_f + V_{SB}} - \sqrt{2\phi_f})$$

    This allows the model to "understand" the Body Effect and accurately classify operating regions dynamically.

### GAN Architecture

-   **Generator:** Fully connected network with Batch Normalization and LeakyReLU.

-   **Discriminator:** Binary classifier estimating the probability of a sample being "real."

-   **Constraint:** GAN outputs are post-processed to ensure positive physical values (e.g., Width $> 0$).

* * * * *

Results
----------

| **Metric** | **Baseline (Standard MLP)** | **This Work (SimpleNN + Physics Features)** |
| --- | --- | --- |
| **MAPE** | ~11.16% | **~7.69%** |
| **R²** | 0.98 | **>0.999** |

*Note: Results may vary slightly depending on the random seed set in `configs/baseline.yaml`.*

* * * * *

License
----------

This project is licensed under the MIT License - see the LICENSE file for details.

Author: Rüya Sanver

Institution: Sabancı University (PURE Project)
