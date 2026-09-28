import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import NEWS_ALIGNMENT_CSV, NEWS_CLEANED_CSV, RANDOM_SEED, SVD_COMPONENTS, TFIDF_PARAMS


def build_daily_text(text_source: str) -> pd.DataFrame:
    if text_source not in {"title", "body", "title+body"}:
        raise ValueError(f"text_source tidak dikenal: {text_source}")
    news = pd.read_csv(NEWS_CLEANED_CSV, usecols=["article_id", "title_clean", "content_clean"])
    alignment = pd.read_csv(
        NEWS_ALIGNMENT_CSV, usecols=["article_id", "effective_trading_date"], parse_dates=["effective_trading_date"]
    )
    merged = alignment.merge(news, on="article_id", how="inner")

    # Dokumen harian hanya dari berita yang sudah terbit sebelum hari perdagangan t
    # (alignment next trading day), sehingga TF-IDF tidak melihat teks masa depan.
    if text_source == "title":
        text = merged["title_clean"].fillna("")
    elif text_source == "body":
        text = merged["content_clean"].fillna("")
    else:
        text = merged["title_clean"].fillna("") + "\n" + merged["content_clean"].fillna("")
    merged["text"] = text

    daily = merged.groupby("effective_trading_date")["text"].apply(lambda parts: "\n".join(parts))
    daily = daily.rename_axis("trading_date").reset_index().rename(columns={"text": "daily_text"})
    return daily


def build_text_pipeline() -> Pipeline:
    # TF-IDF + SVD punya state terlatih: selalu fit di dalam pipeline (train only),
    # bukan pada seluruh dataset sebelum split.
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("svd", TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_SEED)),
            ("scaler", StandardScaler()),
        ]
    )
