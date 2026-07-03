import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import fbeta_score, precision_recall_curve, auc
from hmmlearn import hmm
import xgboost as xgb
import optuna


class DataPreprocessor:
    """Handles feature engineering for the 15-minute telemetry data."""

    def __init__(self):
        # Using Yeo-Johnson to handle right-skewed hydrological variables
        self.pt = PowerTransformer(method='yeo-johnson')

    def add_cyclical_features(self, df, hour_col='hour'):
        """Encodes time of day as sine/cosine pairs to capture wildfowl/tidal rhythms."""
        df = df.copy()
        df['hour_sin'] = np.sin(2 * np.pi * df[hour_col] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df[hour_col] / 24.0)
        return df

    def fit_transform(self, X_telemetry):
        """Fits the Yeo-Johnson transformer and returns normalized features."""
        return self.pt.fit_transform(X_telemetry)

    def transform(self, X_telemetry):
        return self.pt.transform(X_telemetry)


class StormSeqHMM:
    """Layer 1: STORM-SEQ unsupervised ecological phase segmentation."""

    def __init__(self, n_states=5):
        # K=5 latent states representing the pollution lifecycle
        self.n_states = n_states
        # Initializing the model for Baum-Welch EM algorithm training
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100
        )

    def fit(self, X_unlabelled):
        """Trains the HMM on the full unlabelled 15-minute telemetry record."""
        self.model.fit(X_unlabelled)

    def predict_phase(self, X_current):
        """Assigns the current telemetry observation to one of the 5 phases."""
        return self.model.predict(X_current)


class PrismMatrix:
    """Layer 2: PRISM real-time sampling directive matrix."""

    def __init__(self):
        self.risk_median = None
        self.info_median = None

    def calibrate_thresholds(self, historical_risk_scores, historical_info_scores):
        """Sets High/Low thresholds at the 50th percentile of empirical distributions."""
        self.risk_median = np.median(historical_risk_scores)
        self.info_median = np.median(historical_info_scores)

    def calculate_information_gain(self, X_current, X_training_set):
        """Quantifies novelty using nearest neighbor distance in feature space."""
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_training_set)
        distance, _ = nn.kneighbors(X_current)
        return distance

    def get_directive(self, risk_score, info_score):
        """
        Returns a sampling directive based on the 2x2 PRISM decision matrix.

        Matrix:
          High Risk / High Info  -> PRIORITY SAMPLE (immediate intensive sampling)
          High Risk / Low Info   -> RISK SAMPLE     (sample to confirm risk)
          Low Risk  / High Info  -> INFO SAMPLE     (opportunistic data collection)
          Low Risk  / Low Info   -> ROUTINE          (standard monitoring interval)
        """
        high_risk = risk_score >= self.risk_median
        high_info = info_score >= self.info_median

        if high_risk and high_info:
            return "PRIORITY_SAMPLE"
        elif high_risk and not high_info:
            return "RISK_SAMPLE"
        elif not high_risk and high_info:
            return "INFO_SAMPLE"
        else:
            return "ROUTINE"


class ModelPipeline:
    """Orchestrates LOOCV evaluation and Optuna hyperparameter optimisation."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def loocv_evaluate(self, params):
        """Runs Leave-One-Out Cross-Validation for a given set of XGBoost params."""
        loo = LeaveOneOut()
        y_true_all, y_prob_all = [], []

        for train_idx, test_idx in loo.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            model = xgb.XGBClassifier(
                **params,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[0][1]
            y_true_all.append(y_test[0])
            y_prob_all.append(prob)

        # Optimise threshold for F2 score (recall-weighted)
        precision, recall, thresholds = precision_recall_curve(y_true_all, y_prob_all)
        f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-9)
        best_threshold = thresholds[np.argmax(f2_scores)]
        y_pred = (np.array(y_prob_all) >= best_threshold).astype(int)
        return fbeta_score(y_true_all, y_pred, beta=2)

    def optuna_objective(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20),
        }
        return self.loocv_evaluate(params)

    def run_optimisation(self, n_trials=50):
        """Runs Optuna study to find best XGBoost hyperparameters via F2 score."""
        study = optuna.create_study(direction='maximize')
        study.optimize(self.optuna_objective, n_trials=n_trials)
        print(f"Best F2 score: {study.best_value:.4f}")
        print(f"Best params:   {study.best_params}")
        return study.best_params
