"""
Module: train.py
Deskripsi: Orkestrasi eksperimen lengkap 4 Model Baseline & Gabungan untuk Tugas 2:
- Model 0: Market-Only (ARIMAX & XGBoost)
- Model 1: Market + Sentiment (VADER + LM)
- Model 2: Market + TF-IDF (Top 30 n-gram fit on Train)
- Model 3: Full Combined (Market + Sentiment + TF-IDF)

Metrik evaluasi: F1-Score (Macro, Weighted, per-class) dan Confusion Matrix.
Pencegahan data leakage: Seluruh scaler/vectorizer di-fit HANYA pada data Train.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBClassifier

from src.feature_market import extract_market_features, get_market_feature_cols
from src.feature_sentiment import extract_daily_sentiment_features
from src.feature_tfidf import build_daily_tfidf_features
from src.data_preparation import split_chronological
from src.evaluate import compute_metrics, print_evaluation, build_comparison_dataframe


def run_full_experiments(
    news_path: str = "data/cleaned/news_cleaned.csv",
    align_path: str = "data/aligned/news_alignment.csv",
    market_path: str = "data/aligned/daily_aligned_dataset.csv",
    lm_dict_path: str = "data/Loughran-McDonald_MasterDictionary.csv",
    output_dir: str = "data/processed",
    reports_dir: str = "reports",
) -> pd.DataFrame:
    """
    Menjalankan pipeline end-to-end pada seluruh dataset berita dan kurs secara lokal.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    t_start = time.time()
    print("=" * 80)
    print("PIPELINE EKSPERIMEN TUGAS 2: EKSTRAKSI FITUR NLP & MODELING BASELINE")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Ekstraksi Fitur Market
    # -------------------------------------------------------------
    print("\n[LANGKAH 1/4] Ekstraksi Fitur Market-Only...")
    raw_market = pd.read_csv(market_path)
    df_market = extract_market_features(raw_market)
    market_cols = get_market_feature_cols()
    print(f"[OK] Fitur market selesai: {len(df_market)} baris, {len(market_cols)} fitur.")

    # -------------------------------------------------------------
    # 2. Ekstraksi Fitur Sentimen (14.941 Berita)
    # -------------------------------------------------------------
    print(f"\n[LANGKAH 2/4] Ekstraksi Sentimen VADER & Loughran-McDonald...")
    news_df = pd.read_csv(news_path)
    align_df = pd.read_csv(align_path)
    print(f"Total berita yang diproses: {len(news_df):,} artikel.")

    df_sentiment, sentiment_cols = extract_daily_sentiment_features(
        news_df=news_df,
        align_df=align_df,
        dict_path=lm_dict_path,
    )
    print(f"[OK] Fitur sentimen selesai: {len(df_sentiment)} hari bursa, {len(sentiment_cols)} fitur.")

    # -------------------------------------------------------------
    # 3. Ekstraksi Fitur TF-IDF (Fit HANYA pada Train)
    # -------------------------------------------------------------
    print("\n[LANGKAH 3/4] Ekstraksi TF-IDF Harian (Fit on Train Only)...")
    df_tfidf, tfidf_cols, _ = build_daily_tfidf_features(
        news_df=news_df,
        align_df=align_df,
        train_end_date="2024-12-31",
        max_features=30,
    )
    print(f"[OK] Fitur TF-IDF selesai: {len(df_tfidf)} hari bursa, {len(tfidf_cols)} fitur.")

    # -------------------------------------------------------------
    # 4. Penggabungan Fitur & Split Kronologis
    # -------------------------------------------------------------
    print("\n[LANGKAH 4/4] Menggabungkan Seluruh Fitur & Menerapkan Split Kronologis...")
    df_all = df_market.merge(df_sentiment, on="trading_date", how="left")
    df_all = df_all.merge(df_tfidf, on="trading_date", how="left")

    # Impute hari libur/tanpa berita dengan nilai 0 (netral)
    all_feature_cols = market_cols + sentiment_cols + tfidf_cols
    df_all[all_feature_cols] = df_all[all_feature_cols].fillna(0.0)

    train_df, val_df, test_df = split_chronological(df_all)

    # Simpan dataset gabungan yang sudah dipisah
    train_df.to_csv(Path(output_dir) / "train_full.csv", index=False)
    val_df.to_csv(Path(output_dir) / "val_full.csv", index=False)
    test_df.to_csv(Path(output_dir) / "test_full.csv", index=False)

    print(f"Train set      : {len(train_df)} baris ({train_df['trading_date'].min().date()} s/d {train_df['trading_date'].max().date()})")
    print(f"Validation set : {len(val_df)} baris ({val_df['trading_date'].min().date()} s/d {val_df['trading_date'].max().date()})")
    print(f"Test set       : {len(test_df)} baris ({test_df['trading_date'].min().date()} s/d {test_df['trading_date'].max().date()})")

    # -------------------------------------------------------------
    # 5. Pelatihan & Evaluasi 4 Kelompok Model
    # -------------------------------------------------------------
    eval_results = []

    def evaluate_model_on_splits(clf, features, model_label):
        # Fit on train
        clf.fit(train_df[features], train_df["target_up"])
        # Val
        p_val = clf.predict(val_df[features])
        r_val = compute_metrics(val_df["target_up"], p_val, model_label, "Validation")
        eval_results.append(r_val)
        print_evaluation(r_val)
        # Test
        p_test = clf.predict(test_df[features])
        r_test = compute_metrics(test_df["target_up"], p_test, model_label, "Test")
        eval_results.append(r_test)
        print_evaluation(r_test)

    # --- MODEL 0: Market-Only ---
    print("\n" + "=" * 50)
    print("MODEL 0: MARKET-ONLY (ARIMAX & XGBoost)")
    print("=" * 50)
    # 0a. ARIMAX
    arimax_exog = ["return_lag1", "volatility_5d", "sma5_ratio", "momentum_5d"]
    model_arimax = SARIMAX(
        endog=train_df["next_return"],
        exog=train_df[arimax_exog],
        order=(1, 0, 1),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    p_val_arimax = (model_arimax.apply(val_df["next_return"], exog=val_df[arimax_exog], refit=False).fittedvalues > 0).astype(int)
    r_val_arimax = compute_metrics(val_df["target_up"], p_val_arimax, "Model 0 - ARIMAX (Market)", "Validation")
    eval_results.append(r_val_arimax)
    print_evaluation(r_val_arimax)

    p_test_arimax = (model_arimax.apply(test_df["next_return"], exog=test_df[arimax_exog], refit=False).fittedvalues > 0).astype(int)
    r_test_arimax = compute_metrics(test_df["target_up"], p_test_arimax, "Model 0 - ARIMAX (Market)", "Test")
    eval_results.append(r_test_arimax)
    print_evaluation(r_test_arimax)

    # 0b. XGBoost
    xgb_m0 = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss")
    evaluate_model_on_splits(xgb_m0, market_cols, "Model 0 - XGBoost (Market)")

    # --- MODEL 1: Market + Sentiment (VADER + LM) ---
    print("\n" + "=" * 50)
    print("MODEL 1: MARKET + SENTIMENT (VADER + LM)")
    print("=" * 50)
    m1_cols = market_cols + sentiment_cols
    xgb_m1 = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss")
    evaluate_model_on_splits(xgb_m1, m1_cols, "Model 1 - XGBoost (Market + Sentiment)")

    # --- MODEL 2: Market + TF-IDF ---
    print("\n" + "=" * 50)
    print("MODEL 2: MARKET + TF-IDF")
    print("=" * 50)
    m2_cols = market_cols + tfidf_cols
    xgb_m2 = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss")
    evaluate_model_on_splits(xgb_m2, m2_cols, "Model 2 - XGBoost (Market + TF-IDF)")

    # --- MODEL 3: Full Combined (Market + Sentiment + TF-IDF) ---
    print("\n" + "=" * 50)
    print("MODEL 3: FULL COMBINED (Market + Sentiment + TF-IDF)")
    print("=" * 50)
    m3_cols = market_cols + sentiment_cols + tfidf_cols
    xgb_m3 = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss")
    evaluate_model_on_splits(xgb_m3, m3_cols, "Model 3 - XGBoost (Full Combined)")

    # -------------------------------------------------------------
    # 6. Komparasi Akhir & Penyimpanan Laporan
    # -------------------------------------------------------------
    summary_df = build_comparison_dataframe(eval_results)
    summary_path = Path(reports_dir) / "experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    t_end = time.time()
    print("\n" + "=" * 80)
    print("TABEL RINGKASAN HASIL EKSPERIMEN LENGKAP (F1-SCORE & CONFUSION MATRIX)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\n[OK] Waktu eksekusi total: {t_end - t_start:.2f} detik")
    print(f"[OK] Tabel hasil tersimpan di: {summary_path}")

    return summary_df


if __name__ == "__main__":
    run_full_experiments()
