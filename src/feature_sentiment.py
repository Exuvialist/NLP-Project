from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pysentiment2 as ps
from pysentiment2.utils import Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import NEWS_ALIGNMENT_CSV, NEWS_CLEANED_CSV, SENTIMENT_CSV

LM_CATEGORIES = ["Negative", "Positive", "Uncertainty", "Modal"]
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05


def stem_first(tokenizer: Tokenizer, word: str) -> str | None:
    tokens = tokenizer.tokenize(str(word))
    return tokens[0] if tokens else None


def load_lm_category_sets(tokenizer: Tokenizer) -> dict[str, set[str]]:
    data = pd.read_csv(Path(ps.__file__).resolve().parent / "static" / "LM.csv", usecols=["Word"] + LM_CATEGORIES)
    category_sets = {}
    for category in LM_CATEGORIES:
        words = data.loc[data[category] > 0, "Word"].apply(lambda w: stem_first(tokenizer, w)).dropna()
        category_sets[category] = set(words)
    return category_sets


def vader_article_score(sia: SentimentIntensityAnalyzer, title: str, content: str) -> float:
    paragraphs = [str(title)] + [p for p in str(content).split("\n") if p.strip()]
    compounds = [sia.polarity_scores(p)["compound"] for p in paragraphs]
    return float(np.mean(compounds)) if compounds else 0.0


def lm_article_scores(tokenizer: Tokenizer, category_sets: dict[str, set[str]], title: str, content: str) -> dict[str, float]:
    tokens = tokenizer.tokenize(str(title) + "\n" + str(content))
    n_tokens = len(tokens)
    counts = {category: 0 for category in LM_CATEGORIES}
    for token in tokens:
        for category, words in category_sets.items():
            if token in words:
                counts[category] += 1
    per_1k = 1000.0 / max(n_tokens, 1)
    pos = counts["Positive"]
    neg = counts["Negative"]
    return {
        "lm_polarity": (pos - neg) / (pos + neg + 1.0),
        "lm_negative": neg * per_1k,
        "lm_positive": pos * per_1k,
        "lm_uncertainty": counts["Uncertainty"] * per_1k,
        "lm_modal": counts["Modal"] * per_1k,
        "lm_tokens": float(n_tokens),
    }


def score_articles(force: bool = False) -> pd.DataFrame:
    if SENTIMENT_CSV.exists() and not force:
        return pd.read_csv(SENTIMENT_CSV)
    print("skor sentimen artikel (VADER + LM, teks title+body): mulai")
    news = pd.read_csv(NEWS_CLEANED_CSV, usecols=["article_id", "title_clean", "content_clean"])
    sia = SentimentIntensityAnalyzer()
    tokenizer = Tokenizer()
    category_sets = load_lm_category_sets(tokenizer)
    rows = []
    for i, row in enumerate(news.itertuples(index=False), 1):
        scores = {"article_id": row.article_id}
        scores["vader_score"] = vader_article_score(sia, row.title_clean, row.content_clean)
        scores.update(lm_article_scores(tokenizer, category_sets, row.title_clean, row.content_clean))
        rows.append(scores)
        if i % 2000 == 0:
            print(f"  {i}/{len(news)} artikel")
    out = pd.DataFrame(rows)
    out.to_csv(SENTIMENT_CSV, index=False)
    print(f"skor sentimen artikel: {len(out)} baris -> {SENTIMENT_CSV.name}")
    return out


def build_daily_sentiment(force: bool = False) -> pd.DataFrame:
    alignment = pd.read_csv(
        NEWS_ALIGNMENT_CSV, usecols=["article_id", "effective_trading_date"], parse_dates=["effective_trading_date"]
    )
    scores = score_articles(force=force)
    merged = alignment.merge(scores, on="article_id", how="inner")

    # Cutoff konservatif: berita hanya dipakai untuk hari perdagangan SETELAH tanggal
    # publikasi WIB (aturan next trading day Tugas 1), jadi tidak ada look-ahead.
    merged["vader_pos"] = (merged["vader_score"] > VADER_POS_THRESHOLD).astype(float)
    merged["vader_neg"] = (merged["vader_score"] < VADER_NEG_THRESHOLD).astype(float)
    merged["vader_neutral"] = 1.0 - merged["vader_pos"] - merged["vader_neg"]
    merged["lm_pos"] = (merged["lm_polarity"] > 0).astype(float)
    merged["lm_neg"] = (merged["lm_polarity"] < 0).astype(float)
    merged["lm_neutral"] = 1.0 - merged["lm_pos"] - merged["lm_neg"]

    daily = (
        merged.groupby("effective_trading_date")
        .agg(
            n_articles=("article_id", "size"),
            vader_mean=("vader_score", "mean"),
            vader_median=("vader_score", "median"),
            vader_std=("vader_score", "std"),
            vader_pos_ratio=("vader_pos", "mean"),
            vader_neg_ratio=("vader_neg", "mean"),
            vader_neutral_ratio=("vader_neutral", "mean"),
            lm_polarity_mean=("lm_polarity", "mean"),
            lm_polarity_median=("lm_polarity", "median"),
            lm_polarity_std=("lm_polarity", "std"),
            lm_pos_ratio=("lm_pos", "mean"),
            lm_neg_ratio=("lm_neg", "mean"),
            lm_neutral_ratio=("lm_neutral", "mean"),
            lm_negative_mean=("lm_negative", "mean"),
            lm_positive_mean=("lm_positive", "mean"),
            lm_uncertainty_mean=("lm_uncertainty", "mean"),
            lm_modal_mean=("lm_modal", "mean"),
        )
        .reset_index()
        .rename(columns={"effective_trading_date": "trading_date"})
    )
    return daily
