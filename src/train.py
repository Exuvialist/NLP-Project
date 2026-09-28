from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import data_preparation, feature_tfidf
from src.baseline import naive_predictions
from src.config import (
    LOGISTIC_PARAMS,
    MARKET_FEATURES,
    MODEL_SPECS,
    PREDICTIONS_TEST_CSV,
    PREDICTIONS_VALIDATION_CSV,
    RESULTS_TEXT_SOURCE_CSV,
    SENTIMENT_FEATURES,
    TARGET_COLUMN,
    TEST_CSV,
    TEXT_SOURCES,
    TRAIN_CSV,
    VALIDATION_CSV,
    ensure_output_dirs,
)
from src.evaluate import compute_metrics


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (TRAIN_CSV.exists() and VALIDATION_CSV.exists() and TEST_CSV.exists()):
        data_preparation.main()
    train = pd.read_csv(TRAIN_CSV, parse_dates=["trading_date"])
    validation = pd.read_csv(VALIDATION_CSV, parse_dates=["trading_date"])
    test = pd.read_csv(TEST_CSV, parse_dates=["trading_date"])
    return train, validation, test


def text_frames(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, text_source: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_text = feature_tfidf.build_daily_text(text_source)
    frames = []
    for split in (train, validation, test):
        merged = split.merge(daily_text, on="trading_date", how="left")
        merged["daily_text"] = merged["daily_text"].fillna("")
        frames.append(merged)
    return frames[0], frames[1], frames[2]


def build_estimator(spec: dict) -> Pipeline:
    numeric_features: list[str] = []
    if spec["market"]:
        numeric_features += MARKET_FEATURES
    if spec["sentiment"]:
        numeric_features += SENTIMENT_FEATURES

    transformers = []
    if numeric_features:
        transformers.append(("numeric", StandardScaler(), numeric_features))
    if spec["tfidf"]:
        transformers.append(("tfidf", feature_tfidf.build_text_pipeline(), "daily_text"))
    if not transformers:
        raise ValueError("model tanpa fitur")

    preprocessor = ColumnTransformer(transformers)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("logistic", LogisticRegression(**LOGISTIC_PARAMS)),
        ]
    )


def fit_predict(spec: dict, train: pd.DataFrame, eval_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    estimator = build_estimator(spec)
    estimator.fit(train, train[TARGET_COLUMN])
    proba = estimator.predict_proba(eval_frame)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def append_predictions(store: list[dict], model_name: str, frame: pd.DataFrame, pred: np.ndarray, proba: np.ndarray) -> None:
    for date, y_true, y_pred, y_proba in zip(
        frame["trading_date"].dt.strftime("%Y-%m-%d"), frame[TARGET_COLUMN], pred, proba
    ):
        store.append(
            {
                "model": model_name,
                "trading_date": date,
                "y_true": int(y_true),
                "y_pred": int(y_pred),
                "y_proba": float(y_proba),
            }
        )


def select_text_source(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> str:
    rows = []
    for source in TEXT_SOURCES:
        train_text, validation_text, _ = text_frames(train, validation, test, source)
        pred, proba = fit_predict(MODEL_SPECS["model3_market_sentiment_tfidf"], train_text, validation_text)
        rows.append({"text_source": source, **compute_metrics(validation_text[TARGET_COLUMN], pred, proba)})
    results = pd.DataFrame(rows).sort_values(["f1_macro", "directional_accuracy"], ascending=False).reset_index(drop=True)
    results.to_csv(RESULTS_TEXT_SOURCE_CSV, index=False)
    print("pemilihan text source TF-IDF (validasi 2025, model3):")
    print(results.round(4).to_string(index=False))
    return str(results.loc[0, "text_source"])


def run_models(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, text_source: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_text, validation_text, test_text = text_frames(train, validation, test, text_source)
    predictions_validation: list[dict] = []
    predictions_test: list[dict] = []

    for model_name, spec in MODEL_SPECS.items():
        pred_validation, proba_validation = fit_predict(spec, train_text, validation_text)
        append_predictions(predictions_validation, model_name, validation_text, pred_validation, proba_validation)
        pred_test, proba_test = fit_predict(spec, train_text, test_text)
        append_predictions(predictions_test, model_name, test_text, pred_test, proba_test)

    for naive_name, pred in naive_predictions(validation_text).items():
        append_predictions(predictions_validation, naive_name, validation_text, pred, pred.astype(float))
    for naive_name, pred in naive_predictions(test_text).items():
        append_predictions(predictions_test, naive_name, test_text, pred, pred.astype(float))

    return pd.DataFrame(predictions_validation), pd.DataFrame(predictions_test)


def main() -> str:
    ensure_output_dirs()
    train, validation, test = load_splits()
    print(
        f"split: train {len(train)} ({train['trading_date'].min().date()}..{train['trading_date'].max().date()}) | "
        f"validation {len(validation)} ({validation['trading_date'].min().date()}..{validation['trading_date'].max().date()}) | "
        f"test {len(test)} ({test['trading_date'].min().date()}..{test['trading_date'].max().date()})"
    )
    text_source = select_text_source(train, validation, test)
    print(f"text source terpilih untuk laporan akhir: {text_source}")

    predictions_validation, predictions_test = run_models(train, validation, test, text_source)
    predictions_validation.to_csv(PREDICTIONS_VALIDATION_CSV, index=False)
    predictions_test.to_csv(PREDICTIONS_TEST_CSV, index=False)

    for name, predictions in [("validation", predictions_validation), ("test", predictions_test)]:
        summary = pd.DataFrame(
            [
                {"model": model, "n": len(group), **compute_metrics(group["y_true"], group["y_pred"], group["y_proba"])}
                for model, group in predictions.groupby("model", sort=False)
            ]
        )
        # Baseline naif tetap dihitung dan tersimpan di CSV prediksi, tetapi
        # tidak ditampilkan di tabel ringkasan agar fokus pada model0-model3.
        summary_models = summary[summary["model"].str.startswith("model")]
        print(f"ringkasan metrik {name} (model0-model3; baseline naif dirujuk di laporan):")
        print(summary_models.round(4).to_string(index=False))
    return text_source


if __name__ == "__main__":
    main()
