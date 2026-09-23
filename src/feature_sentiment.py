"""
Module: feature_sentiment.py
Deskripsi: Ekstraksi fitur sentimen teks berita geopolitik menggunakan dua pendekatan lexicon:
1. VADER (Valence Aware Dictionary and sEntiment Reasoner) - Lexicon umum berbasis aturan
2. Loughran-McDonald (LM) - Lexicon khusus domain finansial dan ekonomi (10-K filings)
Dilengkapi fungsi agregasi harian untuk dipetakan ke hari bursa.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Pastikan vader_lexicon tersedia secara lokal
try:
    _ = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


class LoughranMcDonaldAnalyzer:
    """
    Penganalisis sentimen berbasis kamus finansial Loughran-McDonald (LM).
    Mengkategorikan kata ke dalam kategori Positive, Negative, dan Uncertainty.
    """

    def __init__(self, dict_path: str = "data/Loughran-McDonald_MasterDictionary.csv"):
        self.dict_path = dict_path
        self.pos_words: Set[str] = set()
        self.neg_words: Set[str] = set()
        self.unc_words: Set[str] = set()
        self._load_dictionary()

    def _load_dictionary(self):
        """Memuat daftar kata dari berkas Master Dictionary Loughran-McDonald."""
        if not os.path.exists(self.dict_path):
            raise FileNotFoundError(
                f"Berkas kamus LM tidak ditemukan di {self.dict_path}. "
                "Silakan pastikan path file benar."
            )

        df = pd.read_csv(self.dict_path, usecols=["Word", "Positive", "Negative", "Uncertainty"])
        df["Word"] = df["Word"].astype(str).str.upper()

        self.pos_words = set(df[df["Positive"] > 0]["Word"])
        self.neg_words = set(df[df["Negative"] > 0]["Word"])
        self.unc_words = set(df[df["Uncertainty"] > 0]["Word"])

    def score_text(self, text: str) -> Dict[str, Any]:
        """
        Menghitung frekuensi kata sentimen dan skor polaritas LM.
        Polaritas = (Positive - Negative) / (Positive + Negative)
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "pos_count": 0,
                "neg_count": 0,
                "unc_count": 0,
                "total_words": 0,
                "polarity": 0.0,
                "sentiment_ratio": 0.0,
                "unc_ratio": 0.0,
                "pos_ratio": 0.0,
                "neg_ratio": 0.0,
            }

        tokens = re.findall(r"[A-Za-z]+", text.upper())
        total_words = len(tokens)
        if total_words == 0:
            return {
                "pos_count": 0,
                "neg_count": 0,
                "unc_count": 0,
                "total_words": 0,
                "polarity": 0.0,
                "sentiment_ratio": 0.0,
                "unc_ratio": 0.0,
                "pos_ratio": 0.0,
                "neg_ratio": 0.0,
            }

        pos_matches = [w for w in tokens if w in self.pos_words]
        neg_matches = [w for w in tokens if w in self.neg_words]
        unc_matches = [w for w in tokens if w in self.unc_words]

        p = len(pos_matches)
        n = len(neg_matches)
        u = len(unc_matches)

        polarity = (p - n) / (p + n) if (p + n) > 0 else 0.0
        sentiment_ratio = (p - n) / total_words
        unc_ratio = u / total_words
        pos_ratio = p / total_words
        neg_ratio = n / total_words

        return {
            "pos_count": p,
            "neg_count": n,
            "unc_count": u,
            "total_words": total_words,
            "polarity": polarity,
            "sentiment_ratio": sentiment_ratio,
            "unc_ratio": unc_ratio,
            "pos_ratio": pos_ratio,
            "neg_ratio": neg_ratio,
        }


class VADERAnalyzer:
    """
    Wrapper untuk VADER Sentiment Intensity Analyzer.
    """

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Menghasilkan skor polaritas VADER: compound, pos, neg, neu.
        """
        if not isinstance(text, str) or not text.strip():
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        return self.sia.polarity_scores(text)


def extract_daily_sentiment_features(
    news_df: pd.DataFrame,
    align_df: pd.DataFrame,
    dict_path: str = "data/Loughran-McDonald_MasterDictionary.csv",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Mengekstrak sentimen VADER pada judul berita dan Loughran-McDonald pada isi berita
    untuk seluruh artikel, lalu melakukan agregasi per hari bursa efektif.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        (df_daily_sentiment, list_kolom_fitur_sentimen)
    """
    print(f"Menghubungkan {len(news_df):,} artikel dengan tanggal bursa...")
    merged = news_df.merge(
        align_df[["article_id", "effective_trading_date"]],
        on="article_id",
        how="inner",
    )

    vader = VADERAnalyzer()
    lm = LoughranMcDonaldAnalyzer(dict_path=dict_path)

    print("Mengekstrak VADER compound pada seluruh judul berita...")
    merged["vader_compound"] = merged["title_clean"].apply(
        lambda x: vader.score_text(str(x))["compound"]
    )
    merged["vader_pos"] = merged["title_clean"].apply(
        lambda x: vader.score_text(str(x))["pos"]
    )
    merged["vader_neg"] = merged["title_clean"].apply(
        lambda x: vader.score_text(str(x))["neg"]
    )

    print("Mengekstrak Loughran-McDonald polaritas & uncertainty pada isi berita...")
    lm_results = merged["content_clean"].apply(
        lambda x: lm.score_text(str(x)[:2500])
    )
    merged["lm_polarity"] = lm_results.apply(lambda r: r["polarity"])
    merged["lm_unc_ratio"] = lm_results.apply(lambda r: r["unc_ratio"])
    merged["lm_pos_ratio"] = lm_results.apply(lambda r: r["pos_ratio"])
    merged["lm_neg_ratio"] = lm_results.apply(lambda r: r["neg_ratio"])

    print("Mengagregasi sentimen per hari bursa...")
    daily_sent = (
        merged.groupby("effective_trading_date")
        .agg(
            vader_compound_mean=("vader_compound", "mean"),
            vader_compound_min=("vader_compound", "min"),
            vader_compound_max=("vader_compound", "max"),
            vader_compound_std=("vader_compound", "std"),
            vader_pos_mean=("vader_pos", "mean"),
            vader_neg_mean=("vader_neg", "mean"),
            lm_polarity_mean=("lm_polarity", "mean"),
            lm_polarity_min=("lm_polarity", "min"),
            lm_polarity_max=("lm_polarity", "max"),
            lm_uncertainty_mean=("lm_unc_ratio", "mean"),
            lm_pos_ratio_mean=("lm_pos_ratio", "mean"),
            lm_neg_ratio_mean=("lm_neg_ratio", "mean"),
            news_volume=("article_id", "count"),
        )
        .reset_index()
    )

    daily_sent.rename(columns={"effective_trading_date": "trading_date"}, inplace=True)
    daily_sent["trading_date"] = pd.to_datetime(daily_sent["trading_date"])
    daily_sent["vader_compound_std"] = daily_sent["vader_compound_std"].fillna(0.0)

    feature_cols = [c for c in daily_sent.columns if c != "trading_date"]

    return daily_sent, feature_cols


def compare_vader_and_lm(
    df_news: pd.DataFrame,
    dict_path: str = "data/Loughran-McDonald_MasterDictionary.csv",
) -> Dict[str, Any]:
    """
    Melakukan perbandingan komprehensif antara VADER dan LM pada teks berita (Title dan Content).
    """
    vader = VADERAnalyzer()
    lm = LoughranMcDonaldAnalyzer(dict_path=dict_path)

    vader_title = df_news["title_clean"].apply(lambda x: vader.score_text(str(x))["compound"])
    lm_title_res = df_news["title_clean"].apply(lambda x: lm.score_text(str(x)))
    lm_title_pol = lm_title_res.apply(lambda r: r["polarity"])

    vader_content = df_news["content_clean"].apply(
        lambda x: vader.score_text(str(x)[:2000])["compound"]
    )
    lm_content_res = df_news["content_clean"].apply(
        lambda x: lm.score_text(str(x)[:2000])
    )
    lm_content_pol = lm_content_res.apply(lambda r: r["polarity"])

    n_total = len(df_news)

    summary = {
        "n_samples": n_total,
        "title_coverage": {
            "vader_nonzero": int((vader_title != 0).sum()),
            "vader_nonzero_pct": float((vader_title != 0).mean() * 100),
            "lm_nonzero": int((lm_title_pol != 0).sum()),
            "lm_nonzero_pct": float((lm_title_pol != 0).mean() * 100),
        },
        "content_coverage": {
            "vader_nonzero": int((vader_content != 0).sum()),
            "vader_nonzero_pct": float((vader_content != 0).mean() * 100),
            "lm_nonzero": int((lm_content_pol != 0).sum()),
            "lm_nonzero_pct": float((lm_content_pol != 0).mean() * 100),
        },
        "title_distribution": {
            "vader_pos": float((vader_title > 0.05).mean() * 100),
            "vader_neg": float((vader_title < -0.05).mean() * 100),
            "vader_neu": float(((vader_title >= -0.05) & (vader_title <= 0.05)).mean() * 100),
            "lm_pos": float((lm_title_pol > 0).mean() * 100),
            "lm_neg": float((lm_title_pol < 0).mean() * 100),
            "lm_neu": float((lm_title_pol == 0).mean() * 100),
        },
        "correlation_vader_vs_lm": {
            "title_corr": float(pd.Series(vader_title).corr(pd.Series(lm_title_pol))),
            "content_corr": float(pd.Series(vader_content).corr(pd.Series(lm_content_pol))),
        },
    }

    return summary
