# Few-Shot EEG Direction Detection System

Deep Metric Meta-Learning pipeline for 4-class directional intent decoding (`Left`, `Right`, `Forward`, `Backward`) from multi-channel bipolar EEG signals.

---

## 📁 Project Structure

```
e:\EEG-direction-detection\
├── Documenations\
│   ├── RESEARCH_DOCUMENTATION.md              # Complete theoretical methodology & papers
│   └── DATA_CONVERSION_AND_EXECUTION_GUIDE.md # Operational handbook & protocol
├── Raw data\                                  # Original EEG ASCII text dumps
│   ├── Raw data(Subject 3)\
│   ├── Raw data(Subject Gope)\
│   └── Raw data(Subject Supra)\
├── Stimuli\                                   # Synchronized stimulus video
├── processed_data\                            # Generated structured datasets
│   ├── metadata.csv                           # Master manifest of all trial metadata
│   ├── csv_trials\                            # Human-readable CSV per single trial
│   └── tensors\                               # High-speed PyTorch .npz tensor archives
├── checkpoints\                               # Saved model weights (.pth)
├── results\                                   # Generated plots & evaluation reports
├── src\
│   ├── __init__.py
│   ├── config.py                              # Centralized configuration & hyperparams
│   ├── preprocessing.py                       # Butterworth Bandpass, Notch & Z-score
│   ├── convert_raw_data.py                    # ETL pipeline converting raw text to CSV/NPZ
│   ├── dataset.py                             # PyTorch Episodic (N-way K-shot) Dataset
│   ├── models.py                              # SpatialTemporalProtoNet & EEGNet backbones
│   ├── losses.py                              # Prototypical Loss & Angular Margin Loss
│   ├── train_fewshot.py                       # Meta-learning episodic training engine
│   ├── evaluate.py                            # LOSO-CV cross-validation & visualizer
│   └── inference.py                           # Single-trial real-time predictor
├── requirements.txt                           # Python dependencies
├── run_pipeline.py                            # Master one-click execution pipeline
└── README.md                                  # Project overview & instructions
```

---

## ⚙️ Setup & Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### 1. Convert Raw Data to Structured CSVs and NPZ Tensors
```bash
python -m src.convert_raw_data
```
- Parses all 18 bipolar channels.
- Applies $0.5 - 45\text{ Hz}$ bandpass and $50\text{ Hz}$ notch filtering.
- Slices into $2.0\text{s}$ ($500$ sample) directional trials.
- Exports individual CSVs to `processed_data/csv_trials/` and subject arrays to `processed_data/tensors/`.
- Generates `processed_data/metadata.csv`.

---

### 2. Meta-Learning Training
```bash
python -m src.train_fewshot --k_shot 5 --epochs 80 --model spatial_temporal
```
Options:
- `--k_shot`: Number of calibration support shots per class (default: `5`).
- `--epochs`: Total meta-training epochs (default: `80`).
- `--model`: Backbone architecture (`spatial_temporal` or `eegnet`).

---

### 3. Cross-Validation & Metric Reporting (LOSO-CV)
```bash
python -m src.evaluate
```
- Evaluates meta-learning generalization on unseen target subjects.
- Computes **4-Way Accuracy (%)**, **Cohen's Kappa ($\kappa$)**, and **Information Transfer Rate (ITR)**.
- Saves publication-quality plots to `results/`:
  - `results/fewshot_scaling_curve.png` (Performance across $K=1, 3, 5, 10$ shots).
  - `results/confusion_matrix_k5.png` (Class-level directional confusion matrix).

---

### 4. Single-Trial Test-Time Inference
```python
from src.inference import DirectionPredictor
import numpy as np

predictor = DirectionPredictor(checkpoint_path="checkpoints/protonet_spatial_temporal_5shot.pth")

# Calibrate on target subject using K-shot support trials:
# predictor.calibrate(support_trials, support_labels, k_shot=5)

# Predict single trial from CSV:
direction, confidences = predictor.predict_from_csv("processed_data/csv_trials/sub01_f01_ep001_left.csv")
print(f"Predicted Direction: {direction}")
print(f"Confidences: {confidences}")
```

---

### 5. One-Click Master Pipeline
```bash
python run_pipeline.py --epochs 40 --shots 1 3 5
```
