# Gas Sensor Drift Classification

A PyTorch multiclass classification project using the UCI Gas Sensor Array Drift Dataset. The project trains a baseline neural network to identify six gases from sensor features, then evaluates how well it generalizes to later data batches.

## Project Objective
Gas sensors can change their behavior over time. A model trained on earlier measurements may perform worse on measurements collected later. This project uses a time-ordered train/validation/test split to investigate that problem.

## Dataset
*   **Source:** UCI Gas Sensor Array Drift Dataset
*   **Measurements:** 13,910
*   **Input features:** 128 per measurement
*   **Gas classes:** 6
*   **Temporal batches:** 10

| Label in original data | Gas |
| :--- | :--- |
| 1 | Ethanol |
| 2 | Ethylene |
| 3 | Ammonia |
| 4 | Acetaldehyde |
| 5 | Acetone |
| 6 | Toluene |

*Note: The raw dataset is not stored in this repository. Download it from UCI and configure `data_path` in the notebook to point to the folder containing `batch1.dat` through `batch10.dat`.*

## Method

### Time-Based Split
The batches are kept separate to preserve their temporal order.

| Purpose | Batches | Samples |
| :--- | :--- | :--- |
| Training | B1–B6 | 5,933 |
| Validation | B7 | 3,613 |
| Final test | B8 | 294 |
| Final test | B9 | 470 |
| Final test | B10 | 3,600 |

The scaler is fitted only on the training data (B1–B6) and then applied to validation and test data. Labels are converted from 1–6 to 0–5 for PyTorch classification.

## Baseline Model
The selected baseline is a small multilayer perceptron (MLP):

*   **Architecture:** 128 input features → Linear(128, 64) → ReLU → Linear(64, 6) → class logits
*   **Loss:** Unweighted CrossEntropyLoss
*   **Optimizer:** Adam
*   **Learning rate:** 0.001
*   **Training:** 20 epochs; save the checkpoint with the lowest validation loss

A deeper MLP and class-weighted loss were also explored. They did not improve the overall validation results consistently, so the simple MLP remains the baseline.

## Recorded Test Results
The saved baseline was evaluated on each later batch separately.

| Test batch | Samples | Cross-entropy loss | Accuracy | Macro-F1 |
| :--- | :--- | :--- | :--- | :--- |
| B8 | 294 | 0.6908 | 81.63% | 0.6893 |
| B9 | 470 | 1.1254 | 64.89% | 0.5862 |
| B10 | 3,600 | 1.6441 | 60.94% | 0.5733 |

### Main Observation
Performance is lower on the later test batches. Acetaldehyde is a particular weakness: its recall was 15.2% on B8, 20.0% on B9, and 3.8% on B10. Toluene recall was 0% on B9. 

These results are consistent with temporal drift or changes in feature/class distributions, but they do not prove that sensor drift is the only cause. B8 and B9 are relatively small, so their per-class metrics should be interpreted cautiously.

## Repository Structure

```text
ANN/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── model.py
├── models/
│   ├── baseline_best.pth
│   └── scaler.pkl
├── eda.ipynb
├── training.ipynb
├── evaluation.ipynb
├── requirements.txt
└── README.md
```

*   `baseline_best.pth`: Contains the selected model weights.
*   `scaler.pkl`: Contains the fitted preprocessing scaler.
*   `src/model.py`: Defines the architecture.

## Run the Project

1. **Install dependencies:** Clone this repository and install the required packages.
   ```bash
   pip install -r requirements.txt
   ```
2. **Download data:** Download the UCI dataset and point `data_path` in the notebook to the folder containing the ten `.dat` batches.
3. **Explore:** Run `eda.ipynb` to explore the dataset.
4. **Train:** Run `training.ipynb` to preprocess the data, train the baseline, evaluate on validation batch B7, and save the best checkpoint and scaler.
5. **Evaluate:** Run `evaluation.ipynb` to load the saved artifacts and evaluate B8, B9, and B10 separately.

*Note: The notebooks may contain environment-specific paths (for example, Google Colab/Drive paths); update those paths for your own environment. Do not place the raw dataset in Git or commit it to this repository.*

## Limitations and Future Work
*   This is a baseline model; performance on later batches degrades.
*   The results suggest a temporal generalization problem, but do not isolate its exact cause.
*   A future experiment could test drift compensation/domain adaptation (for example, CORAL) against this fixed baseline. CORAL has not been implemented or evaluated in the current results.