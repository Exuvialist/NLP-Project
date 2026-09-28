from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
ALIGNED_DIR = DATA_DIR / "aligned"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

NEWS_CLEANED_CSV = CLEANED_DIR / "news_cleaned.csv"
NEWS_ALIGNMENT_CSV = ALIGNED_DIR / "news_alignment.csv"
DAILY_ALIGNED_CSV = ALIGNED_DIR / "daily_aligned_dataset.csv"
SENTIMENT_CSV = CLEANED_DIR / "news_sentiment.csv"

TRAIN_CSV = PROCESSED_DIR / "train.csv"
VALIDATION_CSV = PROCESSED_DIR / "validation.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
AUDIT_JSON = PROCESSED_DIR / "audit_summary.json"

PREDICTIONS_VALIDATION_CSV = REPORTS_DIR / "predictions_validation.csv"
PREDICTIONS_TEST_CSV = REPORTS_DIR / "predictions_test.csv"
RESULTS_VALIDATION_CSV = REPORTS_DIR / "results_validation.csv"
RESULTS_TEST_CSV = REPORTS_DIR / "results_test.csv"
RESULTS_TEXT_SOURCE_CSV = REPORTS_DIR / "results_text_source.csv"

RANDOM_SEED = 42
TARGET_COLUMN = "target_next_up"

TRAIN_END = "2024-12-31"
VALIDATION_END = "2025-12-31"

DEFAULT_TEXT_SOURCE = "title+body"
TEXT_SOURCES = ("title", "body", "title+body")

TFIDF_PARAMS = {
    "max_features": 5000,
    "min_df": 3,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
}
SVD_COMPONENTS = 50
LOGISTIC_PARAMS = {"max_iter": 5000, "random_state": RANDOM_SEED}

MARKET_FEATURES = [
    "ret_1d",
    "ret_2d",
    "ret_3d",
    "ret_5d",
    "rate_to_ma5",
    "rate_to_ma20",
    "vol_5",
    "vol_20",
    "log_rate",
]

SENTIMENT_FEATURES = [
    "n_articles",
    "vader_mean",
    "vader_median",
    "vader_std",
    "vader_pos_ratio",
    "vader_neg_ratio",
    "vader_neutral_ratio",
    "lm_polarity_mean",
    "lm_polarity_median",
    "lm_polarity_std",
    "lm_pos_ratio",
    "lm_neg_ratio",
    "lm_neutral_ratio",
    "lm_negative_mean",
    "lm_positive_mean",
    "lm_uncertainty_mean",
    "lm_modal_mean",
]

FORBIDDEN_COLUMNS = {
    "rate_change",
    "daily_return",
    "has_geopolitical_news",
    "article_ids",
    "usd_idr_rate_next",
}

MODEL_SPECS = {
    "model0_market": {"market": True, "sentiment": False, "tfidf": False},
    "model1_market_sentiment": {"market": True, "sentiment": True, "tfidf": False},
    "model2_market_tfidf": {"market": True, "sentiment": False, "tfidf": True},
    "model3_market_sentiment_tfidf": {"market": True, "sentiment": True, "tfidf": True},
}


def ensure_output_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
