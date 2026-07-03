"""
main.py - Orchestrator script for the STORM-PRISM pipeline.

Usage:
    python src/main.py --data path/to/telemetry.csv --feature-cols col1 col2 col3

This script ties together the DataPreprocessor, StormSeqHMM, and PrismMatrix
classes to run the full two-layer ecological intelligence pipeline.
"""

import argparse
import numpy as np
import pandas as pd

from preprocessor import DataPreprocessor
from storm_seq import StormSeqHMM
from prism import PrismMatrix


def run_pipeline(
    df: pd.DataFrame,
    feature_cols: list,
    hour_col: str = 'hour',
    n_hmm_states: int = 5
) -> pd.DataFrame:
    """
    Runs the full STORM-PRISM pipeline on the provided telemetry DataFrame.

    Args:
        df:            Telemetry DataFrame with a time-of-day column.
        feature_cols:  List of numerical feature column names.
        hour_col:      Column name containing the hour of day (0-23).
        n_hmm_states:  Number of latent states for the HMM (default: 5).

    Returns:
        DataFrame with added columns: 'hmm_phase', 'info_gain', 'directive'.
    """
    # --- Step 1: Preprocessing ---
    print("[1/3] Preprocessing telemetry...")
    preprocessor = DataPreprocessor()
    df = preprocessor.add_cyclical_features(df, hour_col=hour_col)
    all_feature_cols = feature_cols + ['hour_sin', 'hour_cos']
    X_norm = preprocessor.fit_transform(df[all_feature_cols].values)

    # --- Step 2: STORM-SEQ HMM Phase Segmentation ---
    print(f"[2/3] Training STORM-SEQ HMM with {n_hmm_states} states...")
    storm_seq = StormSeqHMM(n_states=n_hmm_states)
    storm_seq.fit(X_norm)
    df['hmm_phase'] = storm_seq.predict_phase(X_norm)
    print(f"      Transition matrix:\n{storm_seq.transition_matrix.round(3)}")

    # --- Step 3: PRISM Directive Matrix ---
    print("[3/3] Calibrating PRISM and computing directives...")
    prism = PrismMatrix()

    # Use HMM phase probability as risk proxy (higher phase index = higher risk)
    risk_scores = df['hmm_phase'].values.astype(float) / n_hmm_states

    # Use nearest-neighbour distance as information gain proxy
    info_scores = np.array([
        prism.calculate_information_gain(X_norm[[i]], X_norm)
        for i in range(len(X_norm))
    ])
    df['info_gain'] = info_scores

    prism.calibrate_thresholds(risk_scores, info_scores)
    df['directive'] = [
        prism.get_directive(r, i)
        for r, i in zip(risk_scores, info_scores)
    ]

    directive_counts = df['directive'].value_counts()
    print(f"\nDirective distribution:\n{directive_counts}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Run the STORM-PRISM pipeline.")
    parser.add_argument('--data', required=True, help='Path to telemetry CSV file.')
    parser.add_argument('--feature-cols', nargs='+', required=True,
                        help='Feature column names to use.')
    parser.add_argument('--hour-col', default='hour',
                        help='Column name for hour of day (default: hour).')
    parser.add_argument('--n-states', type=int, default=5,
                        help='Number of HMM states (default: 5).')
    parser.add_argument('--output', default='output_with_directives.csv',
                        help='Output CSV path.')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} records from {args.data}")

    result_df = run_pipeline(
        df,
        feature_cols=args.feature_cols,
        hour_col=args.hour_col,
        n_hmm_states=args.n_states
    )

    result_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
