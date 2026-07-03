# storm-prism

A two-layer ecological intelligence system for real-time storm overflow detection, developed as part of an MSc Advanced Data Science dissertation at Newcastle University.

## Architecture

The system addresses the **Resolution Mismatch** problem: CSO (Combined Sewer Overflow) events are absolutely rare in 15-minute telemetry records, making standard ML approaches ineffective. The solution decouples the problem into two layers:

### Layer 1 — STORM-SEQ
Unsupervised Hidden Markov Model (5-state Gaussian HMM) that segments the continuous telemetry stream into ecologically meaningful phases (e.g., dry baseline, rising limb, peak storm, recession, recovery). Trained on the full unlabelled record via the Baum-Welch EM algorithm.

### Layer 2 — PRISM
A 2×2 real-time sampling directive matrix that combines a **Risk Score** (from the HMM phase) and an **Information Gain Score** (novelty via nearest-neighbour distance) to issue one of four directives:

| | High Information Gain | Low Information Gain |
|---|---|---|
| **High Risk** | `PRIORITY_SAMPLE` | `RISK_SAMPLE` |
| **Low Risk** | `INFO_SAMPLE` | `ROUTINE` |

## Repository Structure

```
storm-prism/
├── storm_prism_pipeline.py   # Core modular implementation (all 4 classes)
├── src/
│   ├── preprocessor.py       # DataPreprocessor: Yeo-Johnson + cyclical encoding
│   ├── storm_seq.py          # StormSeqHMM: 5-state HMM phase segmentation
│   ├── prism.py              # PrismMatrix: risk/info thresholding & directives
│   └── main.py               # Orchestrator script
├── data/                     # Raw and processed telemetry datasets (gitignored)
├── notebooks/                # Jupyter notebooks for analysis
├── tests/                    # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
git clone https://github.com/fidelmehra/storm-prism.git
cd storm-prism
pip install -r requirements.txt
```

## Usage

```python
from storm_prism_pipeline import DataPreprocessor, StormSeqHMM, PrismMatrix, ModelPipeline

# 1. Preprocess telemetry
preprocessor = DataPreprocessor()
df = preprocessor.add_cyclical_features(df, hour_col='hour')
X_norm = preprocessor.fit_transform(df[feature_cols])

# 2. Train STORM-SEQ HMM
storm_seq = StormSeqHMM(n_states=5)
storm_seq.fit(X_norm)
phases = storm_seq.predict_phase(X_norm)

# 3. Calibrate and query PRISM
prism = PrismMatrix()
prism.calibrate_thresholds(historical_risk, historical_info)
directive = prism.get_directive(current_risk, current_info)
print(directive)  # e.g. 'PRIORITY_SAMPLE'

# 4. Optimise XGBoost via Optuna + LOOCV
pipeline = ModelPipeline(X_labelled, y_labelled)
best_params = pipeline.run_optimisation(n_trials=50)
```

## Dependencies

- `numpy`, `pandas` — data manipulation
- `scikit-learn` — preprocessing, nearest neighbours, cross-validation
- `hmmlearn` — Gaussian HMM
- `xgboost` — gradient boosted classifier
- `optuna` — hyperparameter optimisation
- `matplotlib`, `seaborn` — visualisation

## Context

This system was designed for Northumbrian Water / KKR infrastructure monitoring, targeting the detection of pollution events in ecologically sensitive catchments. The PRISM directive matrix is intended to guide adaptive, resource-efficient field sampling in real time.
