import numpy as np
from hmmlearn import hmm


class StormSeqHMM:
    """Layer 1: STORM-SEQ unsupervised ecological phase segmentation.

    A 5-state Gaussian HMM trained on the full unlabelled 15-minute telemetry
    record to segment the continuous data stream into ecologically meaningful
    phases (e.g., dry baseline, rising limb, peak storm, recession, recovery).
    """

    def __init__(self, n_states: int = 5):
        # K=5 latent states representing the pollution lifecycle
        self.n_states = n_states
        # Initializing the model for Baum-Welch EM algorithm training
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100
        )

    def fit(self, X_unlabelled: np.ndarray) -> None:
        """Trains the HMM on the full unlabelled 15-minute telemetry record."""
        self.model.fit(X_unlabelled)

    def predict_phase(self, X_current: np.ndarray) -> np.ndarray:
        """Assigns the current telemetry observation to one of the K phases."""
        return self.model.predict(X_current)

    def score(self, X: np.ndarray) -> float:
        """Returns the log-likelihood of the sequence under the trained model."""
        return self.model.score(X)

    @property
    def transition_matrix(self) -> np.ndarray:
        """Returns the learned HMM transition probability matrix."""
        return self.model.transmat_

    @property
    def means(self) -> np.ndarray:
        """Returns the emission means for each latent state."""
        return self.model.means_
