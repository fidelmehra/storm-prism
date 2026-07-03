import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple


class PrismMatrix:
    """Layer 2: PRISM real-time sampling directive matrix.

    Combines a Risk Score (from HMM phase probability) and an Information
    Gain Score (novelty via nearest-neighbour distance) to issue one of
    four adaptive sampling directives.

    Decision Matrix:
      High Risk / High Info  -> PRIORITY_SAMPLE  (immediate intensive sampling)
      High Risk / Low Info   -> RISK_SAMPLE       (sample to confirm risk)
      Low Risk  / High Info  -> INFO_SAMPLE       (opportunistic data collection)
      Low Risk  / Low Info   -> ROUTINE            (standard monitoring interval)
    """

    def __init__(self):
        self.risk_median = None
        self.info_median = None

    def calibrate_thresholds(
        self,
        historical_risk_scores: np.ndarray,
        historical_info_scores: np.ndarray
    ) -> None:
        """Sets High/Low thresholds at the 50th percentile of empirical distributions."""
        self.risk_median = np.median(historical_risk_scores)
        self.info_median = np.median(historical_info_scores)

    def calculate_information_gain(
        self,
        X_current: np.ndarray,
        X_training_set: np.ndarray
    ) -> float:
        """Quantifies novelty using nearest-neighbour distance in feature space."""
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_training_set)
        distance, _ = nn.kneighbors(X_current)
        return float(distance.mean())

    def get_directive(self, risk_score: float, info_score: float) -> str:
        """Returns a sampling directive from the 2x2 PRISM decision matrix."""
        if self.risk_median is None or self.info_median is None:
            raise RuntimeError("Thresholds not calibrated. Call calibrate_thresholds() first.")

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

    def get_directive_with_scores(
        self,
        risk_score: float,
        info_score: float
    ) -> Tuple[str, bool, bool]:
        """Returns directive plus boolean flags for high-risk and high-info."""
        directive = self.get_directive(risk_score, info_score)
        high_risk = risk_score >= self.risk_median
        high_info = info_score >= self.info_median
        return directive, high_risk, high_info
