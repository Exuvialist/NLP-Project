import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import LOGISTIC_PARAMS


def build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(**LOGISTIC_PARAMS)),
        ]
    )


def naive_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    # Naive 1: selalu prediksi naik. Naive 2: persistence (arah pergerakan hari sebelumnya).
    return {
        "naive_all_up": np.ones(len(frame), dtype=int),
        "naive_persistence": (frame["ret_1d"].to_numpy() > 0).astype(int),
    }
