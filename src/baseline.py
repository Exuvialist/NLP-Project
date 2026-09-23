"""
Module: baseline.py
Deskripsi: Implementasi model baseline market-only menggunakan ARIMAX dan XGBoost
untuk memprediksi arah pergerakan nilai tukar USD/IDR esok hari.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBClassifier

from src.feature_market import get_market_feature_cols
from src.evaluate import compute_metrics, print_evaluation, build_comparison_dataframe


# Fitur eksogen terpilih untuk ARIMAX (fitur paling relevan tanpa multikolinearitas berlebih)
ARIMAX_EXOG_COLS = [
    "return_lag1",
    "volatility_5d",
    "sma5_ratio",
    "momentum_5d",
]


class ARIMAXBaseline:
    """
    Baseline time-series ARIMAX untuk memprediksi next-day return,
    kemudian dikonversi menjadi arah biner: 1 jika return > 0 (UP), 0 jika <= 0 (DOWN).
    """

    def __init__(self, order: Tuple[int, int, int] = (1, 0, 1), exog_cols: List[str] = None):
        self.order = order
        self.exog_cols = exog_cols or ARIMAX_EXOG_COLS
        self.fitted_model = None

    def fit(self, train_df: pd.DataFrame) -> "ARIMAXBaseline":
        """Melatih model SARIMAX hanya pada data pelatihan."""
        y_train = train_df["next_return"].astype(float)
        X_train = train_df[self.exog_cols].astype(float)

        model = SARIMAX(
            endog=y_train,
            exog=X_train,
            order=self.order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.fitted_model = model.fit(disp=False)
        return self

    def predict_direction(self, df: pd.DataFrame) -> np.ndarray:
        """
        Melakukan prediksi 1-step walk forward tanpa data leakage
        menggunakan kalman filter (results.apply) pada data evaluasi.
        """
        if self.fitted_model is None:
            raise ValueError("Model belum di-fit!")

        y_eval = df["next_return"].astype(float)
        X_eval = df[self.exog_cols].astype(float)

        # Menerapkan parameter hasil fit ke data evaluasi secara sekuensial (zero-leakage)
        eval_results = self.fitted_model.apply(y_eval, exog=X_eval, refit=False)
        pred_returns = eval_results.fittedvalues.values

        # Konversi ke arah biner: > 0 adalah NAIK (1), <= 0 adalah TURUN (0)
        pred_direction = (pred_returns > 0).astype(int)
        return pred_direction


class XGBoostMarketBaseline:
    """
    Baseline Machine Learning XGBoost Classifier yang memprediksi
    arah biner pergerakan kurs esok hari (target_up) berbasis seluruh fitur market.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        feature_cols: List[str] = None,
    ):
        self.feature_cols = feature_cols or get_market_feature_cols()
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric="logloss",
        )

    def fit(self, train_df: pd.DataFrame) -> "XGBoostMarketBaseline":
        X_train = train_df[self.feature_cols]
        y_train = train_df["target_up"]
        self.model.fit(X_train, y_train)
        return self

    def predict_direction(self, df: pd.DataFrame) -> np.ndarray:
        X_eval = df[self.feature_cols]
        return self.model.predict(X_eval)


def run_market_baseline_experiments(
    train_path: str = "data/processed/train_market.csv",
    val_path: str = "data/processed/val_market.csv",
    test_path: str = "data/processed/test_market.csv",
) -> pd.DataFrame:
    """
    Menjalankan pelatihan dan evaluasi kedua model baseline (ARIMAX & XGBoost)
    pada split Validation dan Test set.
    """
    print(f"Memuat data dari:\n  Train : {train_path}\n  Val   : {val_path}\n  Test  : {test_path}")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    eval_results = []

    # ---------------------------------------------------------
    # 1. Model Baseline: ARIMAX
    # ---------------------------------------------------------
    print("\n[1/2] Melatih Model ARIMAX (Market-Only)...")
    arimax = ARIMAXBaseline(order=(1, 0, 1))
    arimax.fit(train_df)

    # Evaluasi ARIMAX pada Validation
    pred_val_arimax = arimax.predict_direction(val_df)
    res_val_arimax = compute_metrics(val_df["target_up"], pred_val_arimax, "ARIMAX", "Validation")
    eval_results.append(res_val_arimax)
    print_evaluation(res_val_arimax)

    # Evaluasi ARIMAX pada Test
    pred_test_arimax = arimax.predict_direction(test_df)
    res_test_arimax = compute_metrics(test_df["target_up"], pred_test_arimax, "ARIMAX", "Test")
    eval_results.append(res_test_arimax)
    print_evaluation(res_test_arimax)

    # ---------------------------------------------------------
    # 2. Model Baseline: XGBoost
    # ---------------------------------------------------------
    print("\n[2/2] Melatih Model XGBoost Classifier (Market-Only)...")
    xgb = XGBoostMarketBaseline(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
    xgb.fit(train_df)

    # Evaluasi XGBoost pada Validation
    pred_val_xgb = xgb.predict_direction(val_df)
    res_val_xgb = compute_metrics(val_df["target_up"], pred_val_xgb, "XGBoost", "Validation")
    eval_results.append(res_val_xgb)
    print_evaluation(res_val_xgb)

    # Evaluasi XGBoost pada Test
    pred_test_xgb = xgb.predict_direction(test_df)
    res_test_xgb = compute_metrics(test_df["target_up"], pred_test_xgb, "XGBoost", "Test")
    eval_results.append(res_test_xgb)
    print_evaluation(res_test_xgb)

    # Buat tabel ringkasan komparasi
    summary_df = build_comparison_dataframe(eval_results)
    return summary_df


if __name__ == "__main__":
    summary = run_market_baseline_experiments()
    print("\n=== TABEL PERBANDINGAN METRIK F1 & CONFUSION MATRIX ===")
    print(summary.to_string(index=False))

