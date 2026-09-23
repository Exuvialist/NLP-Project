"""
Module: feature_tfidf.py
Deskripsi: Ekstraksi fitur teks klasik TF-IDF (Term Frequency - Inverse Document Frequency)
pada judul berita geopolitik harian, dengan pencegahan data leakage (fit HANYA pada data Train).
"""

from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def build_daily_tfidf_features(
    news_df: pd.DataFrame,
    align_df: pd.DataFrame,
    train_end_date: str = "2024-12-31",
    max_features: int = 30,
) -> Tuple[pd.DataFrame, List[str], TfidfVectorizer]:
    """
    Menggabungkan seluruh judul berita per hari bursa menjadi satu dokumen harian,
    lalu mengekstrak representasi vektor TF-IDF (unigram & bigram).

    Untuk mencegah data leakage:
    - TfidfVectorizer di-fit HANYA pada dokumen hari bursa Train (s/d train_end_date).
    - Dokumen pada periode Validation dan Test hanya di-transform.

    Parameters
    ----------
    news_df : pd.DataFrame
        DataFrame artikel bersih ('article_id', 'title_clean').
    align_df : pd.DataFrame
        DataFrame pemetaan ('article_id', 'effective_trading_date').
    train_end_date : str
        Batas akhir tanggal data latih (default: "2024-12-31").
    max_features : int
        Jumlah fitur kata teratas yang diekstrak (default: 30).

    Returns
    -------
    Tuple[pd.DataFrame, List[str], TfidfVectorizer]
        (df_tfidf_daily, list_kolom_tfidf, vectorizer)
    """
    # 1. Gabungkan berita dengan tanggal bursa efektif
    merged = news_df[["article_id", "title_clean"]].merge(
        align_df[["article_id", "effective_trading_date"]],
        on="article_id",
        how="inner",
    )

    # 2. Agregasi judul per hari bursa menjadi 1 teks harian
    merged["title_clean"] = merged["title_clean"].fillna("").astype(str)
    daily_titles = (
        merged.groupby("effective_trading_date")["title_clean"]
        .apply(lambda titles: " ".join(titles))
        .reset_index()
    )
    daily_titles.rename(columns={"effective_trading_date": "trading_date"}, inplace=True)
    daily_titles["trading_date"] = pd.to_datetime(daily_titles["trading_date"])
    daily_titles = daily_titles.sort_values("trading_date").reset_index(drop=True)

    # 3. Pisahkan teks Train untuk proses fit
    train_mask = daily_titles["trading_date"] <= pd.to_datetime(train_end_date)
    train_corpus = daily_titles.loc[train_mask, "title_clean"]

    # 4. Inisialisasi dan fit Vectorizer hanya pada Train
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
    )
    tfidf.fit(train_corpus)

    # 5. Transform seluruh teks harian
    tfidf_matrix = tfidf.transform(daily_titles["title_clean"]).toarray()

    # Buat nama kolom bersih
    feature_names = [f"tfidf_{name.replace(' ', '_')}" for name in tfidf.get_feature_names_out()]

    df_tfidf = pd.DataFrame(tfidf_matrix, columns=feature_names)
    df_tfidf.insert(0, "trading_date", daily_titles["trading_date"])

    return df_tfidf, feature_names, tfidf

