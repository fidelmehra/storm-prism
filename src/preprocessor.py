import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer


class DataPreprocessor:
    """Handles feature engineering for the 15-minute telemetry data.

    Applies Yeo-Johnson power transformation to handle right-skewed
    hydrological variables and encodes time of day as cyclical features.
    """

    def __init__(self):
        self.pt = PowerTransformer(method='yeo-johnson')

    def add_cyclical_features(self, df: pd.DataFrame, hour_col: str = 'hour') -> pd.DataFrame:
        """Encodes time of day as sine/cosine pairs to capture wildfowl/tidal rhythms."""
        df = df.copy()
        df['hour_sin'] = np.sin(2 * np.pi * df[hour_col] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df[hour_col] / 24.0)
        return df

    def fit_transform(self, X_telemetry: np.ndarray) -> np.ndarray:
        """Fits the Yeo-Johnson transformer and returns normalized features."""
        return self.pt.fit_transform(X_telemetry)

    def transform(self, X_telemetry: np.ndarray) -> np.ndarray:
        """Transforms new data using the already-fitted transformer."""
        return self.pt.transform(X_telemetry)
