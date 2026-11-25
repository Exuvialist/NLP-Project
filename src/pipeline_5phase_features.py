"""
pipeline_5phase_features.py
===========================
Leakage-free feature engineering helpers for 11_full_5phase_nlp_fusion_pipeline.ipynb:
1. Article-level ProsusAI/finbert sentiment probabilities [pos, neg, neu] on title_clean
2. Dictionary/Gazetteer Makro + FX NER (rule-based domain entity extractor)
3. NER Entity Pre-filtering & Soft Weighting for FinBERT headline pooling (Fase 2)
4. Train-only NMF Topic Modeling (Fase 3, paired with BERTopic(FinBERT))
5. Strictly causal Macro Domain Features: Novelty, Disagreement, Volume Shock (Fase 3)
"""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


GAZETTEER_PATTERNS = {
    "central_bank": re.compile(
        r"\b(fed|federal reserve|fomc|powell|bank indonesia|bi rate|ecb|lagarde|"
        r"boj|ueda|pboc|central bank|rate hike|rate cut|monetary policy|tightening|easing|hawkish|dovish)\b",
        flags=re.IGNORECASE,
    ),
    "currency_fx": re.compile(
        r"\b(usd|idr|rupiah|dollar|greenback|dxy|yen|euro|yuan|rmb|renminbi|"
        r"currency|currencies|forex|fx|exchange rate|depreciat\w*|appreciat\w*)\b",
        flags=re.IGNORECASE,
    ),
    "macro_rates": re.compile(
        r"\b(inflation|cpi|pce|ppi|gdp|unemployment|payrolls|nonfarm|jobs report|"
        r"interest rate|treasury|yield|bond|recession|stagflation|growth|deficit)\b",
        flags=re.IGNORECASE,
    ),
    "geopolitics_trade": re.compile(
        r"\b(tariff\w*|sanction\w*|trade war|war|conflict|middle east|ukraine|russia|"
        r"iran|israel|hormuz|red sea|opec|crude|oil price|brent|export|import|geopolitic\w*)\b",
        flags=re.IGNORECASE,
    ),
    "emerging_indonesia": re.compile(
        r"\b(indonesia\w*|jakarta|asean|emerging market\w*|asian market\w*|asian currenc\w*|"
        r"capital outflow|foreign inflow|rupiah)\b",
        flags=re.IGNORECASE,
    ),
}


def _clean_lexical(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = re.sub(r"<[^>]+>", " ", text.lower())
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_finbert_sentiment_probs(
    df_news: pd.DataFrame,
    cache_path: Path,
    batch_size: int = 128,
    max_length: int = 96,
) -> dict:
    """
    Extracts ProsusAI/finbert 3-class sentiment probabilities [P(pos), P(neg), P(neu)]
    for each unique article title_clean. Cached to disk for instant re-runs.
    """
    cache_path = Path(cache_path)
    ids_path = cache_path.with_name(cache_path.stem + "_ids.npy")

    df_unique = df_news.drop_duplicates("article_id").reset_index(drop=True)
    article_ids = df_unique["article_id"].astype(str).tolist()

    if cache_path.exists() and ids_path.exists():
        cached_ids = np.load(ids_path, allow_pickle=True).tolist()
        if cached_ids == article_ids:
            probs = np.load(cache_path)
            print(f"[FinBERT-Sentiment] Loaded cached probs: {probs.shape}")
            return dict(zip(article_ids, probs))

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FinBERT-Sentiment] Scoring {len(article_ids)} headlines on {device}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
    model.eval()

    # Map model id2label robustly to canonical [positive, negative, neutral] order
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    pos_idx = [k for k, v in id2label.items() if "pos" in v][0]
    neg_idx = [k for k, v in id2label.items() if "neg" in v][0]
    neu_idx = [k for k, v in id2label.items() if "neu" in v][0]

    titles = df_unique["title_clean"].fillna("").astype(str).tolist()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(titles), batch_size):
            batch = titles[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(**enc).logits
                p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            reordered = np.stack([p[:, pos_idx], p[:, neg_idx], p[:, neu_idx]], axis=1)
            all_probs.append(reordered.astype(np.float32))

    probs = np.vstack(all_probs)
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    np.save(cache_path, probs)
    np.save(ids_path, np.array(article_ids, dtype=object))
    print(f"[FinBERT-Sentiment] Done in {time.time() - t0:.2f}s -> {probs.shape}")
    return dict(zip(article_ids, probs))


def build_gazetteer_macro_fx_features(
    merged_news: pd.DataFrame,
    all_trading_dates: np.ndarray,
):
    """
    Rule-based Dictionary/Gazetteer Makro + FX NER on headlines.
    Zero fitted parameters -> strictly zero leakage.
    Returns:
      - daily_gaz_mat: (N_days, 10)
      - article_df: per-article match counts & flags
      - cat_coverage_df: summary table of Gazetteer hit statistics
    """
    t0 = time.time()
    titles = merged_news["title_clean"].fillna("").astype(str).tolist()

    cat_names = list(GAZETTEER_PATTERNS.keys())
    counts_mat = np.zeros((len(titles), len(cat_names)), dtype=np.float32)

    for j, cname in enumerate(cat_names):
        pat = GAZETTEER_PATTERNS[cname]
        counts_mat[:, j] = [len(pat.findall(txt)) for txt in titles]

    total_hits = counts_mat.sum(axis=1, keepdims=True)
    is_macro_relevant = (total_hits[:, 0] > 0).astype(np.float32)
    has_cb = (counts_mat[:, 0] > 0).astype(np.float32)
    has_fx = (counts_mat[:, 1] > 0).astype(np.float32)
    has_macro = (counts_mat[:, 2] > 0).astype(np.float32)
    has_geo = (counts_mat[:, 3] > 0).astype(np.float32)

    # 10 features per article before daily mean aggregation
    art_feats = np.column_stack([
        counts_mat,          # 5 category counts
        has_cb,              # ratio mentioning central banks
        has_fx,              # ratio mentioning FX/currencies
        has_macro,           # ratio mentioning macro indicators
        has_geo,             # ratio mentioning geopolitics/trade
        is_macro_relevant,   # overall macro/FX relevance ratio
    ]).astype(np.float32)

    df_tmp = pd.DataFrame(art_feats)
    df_tmp["effective_trading_date"] = merged_news["effective_trading_date"].values
    daily_map = df_tmp.groupby("effective_trading_date").mean().to_dict(orient="index")

    daily_gaz_mat = np.array(
        [
            [daily_map[d][c] for c in range(art_feats.shape[1])]
            if d in daily_map
            else np.zeros(art_feats.shape[1])
            for d in all_trading_dates
        ],
        dtype=np.float32,
    )

    art_summary = pd.DataFrame({
        "article_id": merged_news["article_id"].astype(str).values,
        "effective_trading_date": merged_news["effective_trading_date"].values,
        "gaz_total_hits": total_hits[:, 0],
        "gaz_is_relevant": is_macro_relevant,
        "gaz_has_cb": has_cb,
        "gaz_has_fx": has_fx,
        "gaz_has_geo": has_geo,
    })

    cov_rows = []
    for j, cname in enumerate(cat_names):
        n_art_hit = int((counts_mat[:, j] > 0).sum())
        cov_rows.append({
            "Gazetteer_Category": cname,
            "Articles_Matched": n_art_hit,
            "Share_Pct": round(100.0 * n_art_hit / max(1, len(titles)), 2),
            "Total_Mentions": int(counts_mat[:, j].sum()),
        })
    cov_rows.append({
        "Gazetteer_Category": "ANY_MACRO_FX_ENTITY",
        "Articles_Matched": int(is_macro_relevant.sum()),
        "Share_Pct": round(100.0 * is_macro_relevant.mean(), 2),
        "Total_Mentions": int(total_hits.sum()),
    })

    print(
        f"[Gazetteer Makro+FX] Scanned {len(titles)} headlines in {time.time() - t0:.2f}s | "
        f"Macro-relevant coverage: {100.0 * is_macro_relevant.mean():.2f}% -> daily {daily_gaz_mat.shape}"
    )
    return daily_gaz_mat, art_summary, pd.DataFrame(cov_rows)


def build_ner_filtered_finbert_daily(
    merged_news: pd.DataFrame,
    article_emb_map: dict,
    gaz_art_df: pd.DataFrame,
    all_trading_dates: np.ndarray,
):
    """
    Creates FinBERT daily 768d embeddings under different NER pre-filtering / weighting schemes:
      1. finbert_judul_unweighted: standard mean pooling of all articles
      2. finbert_judul_gaz_filtered: keep only Gazetteer Makro+FX headlines (fallback all if 0 on day)
      3. finbert_judul_spacy_filtered: keep only headlines with spaCy macro entities ORG/GPE/NORP/MONEY/PERCENT/EVENT
      4. finbert_judul_ner_weighted: soft weighted mean pooling using spaCy macro flag + Gazetteer hits
    """
    import spacy

    t0 = time.time()
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "attribute_ruler"])
    macro_labels = {"ORG", "GPE", "NORP", "MONEY", "PERCENT", "EVENT"}

    titles = merged_news["title_clean"].fillna("").astype(str).tolist()
    spacy_macro_flag = np.zeros(len(titles), dtype=np.float32)

    for i, doc in enumerate(nlp.pipe(titles, batch_size=256)):
        if any(ent.label_ in macro_labels for ent in doc.ents):
            spacy_macro_flag[i] = 1.0

    aids = merged_news["article_id"].astype(str).tolist()
    embs = np.vstack([article_emb_map[aid] for aid in aids]).astype(np.float32)
    dates = merged_news["effective_trading_date"].values

    gaz_flag = gaz_art_df["gaz_is_relevant"].values.astype(np.float32)
    gaz_hits = np.clip(gaz_art_df["gaz_total_hits"].values.astype(np.float32), 0.0, 3.0)
    soft_weights = (1.0 + 1.0 * spacy_macro_flag + 1.5 * gaz_hits).astype(np.float32)

    date_to_indices = {}
    for idx, d in enumerate(dates):
        date_to_indices.setdefault(d, []).append(idx)

    dim = embs.shape[1]
    n_days = len(all_trading_dates)
    mat_unweighted = np.zeros((n_days, dim), dtype=np.float32)
    mat_gaz_filt = np.zeros((n_days, dim), dtype=np.float32)
    mat_spacy_filt = np.zeros((n_days, dim), dtype=np.float32)
    mat_ner_weighted = np.zeros((n_days, dim), dtype=np.float32)

    fallback_gaz = 0
    fallback_spacy = 0

    for row_idx, d in enumerate(all_trading_dates):
        idxs = date_to_indices.get(d, [])
        if not idxs:
            continue
        idxs_arr = np.array(idxs, dtype=int)
        day_embs = embs[idxs_arr]

        # 1. Unweighted mean
        mat_unweighted[row_idx] = day_embs.mean(axis=0)

        # 2. Gazetteer hard filter with safe fallback
        g_keep = idxs_arr[gaz_flag[idxs_arr] > 0]
        if len(g_keep) > 0:
            mat_gaz_filt[row_idx] = embs[g_keep].mean(axis=0)
        else:
            mat_gaz_filt[row_idx] = mat_unweighted[row_idx]
            fallback_gaz += 1

        # 3. spaCy entity filter with safe fallback
        s_keep = idxs_arr[spacy_macro_flag[idxs_arr] > 0]
        if len(s_keep) > 0:
            mat_spacy_filt[row_idx] = embs[s_keep].mean(axis=0)
        else:
            mat_spacy_filt[row_idx] = mat_unweighted[row_idx]
            fallback_spacy += 1

        # 4. Soft NER-weighted mean pooling
        w = soft_weights[idxs_arr].reshape(-1, 1)
        mat_ner_weighted[row_idx] = (day_embs * w).sum(axis=0) / np.maximum(w.sum(), 1e-6)

    stats = {
        "spacy_macro_article_pct": round(100.0 * float(spacy_macro_flag.mean()), 2),
        "gazetteer_article_pct": round(100.0 * float(gaz_flag.mean()), 2),
        "days_fallback_gaz": fallback_gaz,
        "days_fallback_spacy": fallback_spacy,
    }
    print(
        f"[NER Pre-Filter/Weight] Done in {time.time() - t0:.2f}s | "
        f"spaCy macro cov: {stats['spacy_macro_article_pct']}% (fallback {fallback_spacy}d), "
        f"Gazetteer cov: {stats['gazetteer_article_pct']}% (fallback {fallback_gaz}d)"
    )
    return {
        "unweighted": mat_unweighted,
        "gaz_filtered": mat_gaz_filt,
        "spacy_filtered": mat_spacy_filt,
        "ner_weighted": mat_ner_weighted,
    }, stats


def fit_transform_nmf_daily(
    merged_news: pd.DataFrame,
    all_trading_dates: np.ndarray,
    train_mask: np.ndarray,
    n_topics: int = 15,
    random_state: int = 42,
):
    """
    Fits TF-IDF + NMF strictly on Train headlines, then transforms all articles
    and aggregates L1-normalized topic proportions + topic entropy daily.
    """
    t0 = time.time()
    train_dates = set(all_trading_dates[train_mask])
    is_train = merged_news["effective_trading_date"].isin(train_dates).values

    texts = merged_news["title_clean"].fillna("").map(_clean_lexical).tolist()
    train_texts = [txt for txt, flag in zip(texts, is_train) if flag]

    vec = TfidfVectorizer(
        stop_words="english",
        max_features=3000,
        min_df=3,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_train_tfidf = vec.fit_transform(train_texts)
    X_all_tfidf = vec.transform(texts)

    nmf = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=random_state,
        max_iter=500,
    )
    nmf.fit(X_train_tfidf)
    W_all = np.maximum(nmf.transform(X_all_tfidf), 0.0)

    row_sums = W_all.sum(axis=1, keepdims=True)
    # If row sum is zero (no vocab match), assign uniform 1/K
    zero_rows = row_sums[:, 0] <= 1e-12
    probs = np.divide(W_all, np.where(zero_rows[:, None], 1.0, row_sums))
    probs[zero_rows] = 1.0 / n_topics

    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1, keepdims=True)
    art_features = np.hstack([probs, entropy]).astype(np.float32)

    feat_df = pd.DataFrame(art_features)
    feat_df["effective_trading_date"] = merged_news["effective_trading_date"].values
    daily_map = feat_df.groupby("effective_trading_date").mean().to_dict(orient="index")

    dim = art_features.shape[1]
    daily_mat = np.array(
        [
            [daily_map[d][c] for c in range(dim)] if d in daily_map else np.zeros(dim)
            for d in all_trading_dates
        ],
        dtype=np.float32,
    )

    vocab = np.array(vec.get_feature_names_out())
    topic_rows = []
    for tidx, comp in enumerate(nmf.components_):
        top_ids = comp.argsort()[-10:][::-1]
        topic_rows.append({
            "Model": "NMF",
            "Topic_ID": tidx,
            "Top_Terms": ", ".join(vocab[top_ids]),
        })

    print(f"[NMF] Fit on {is_train.sum()} Train articles ({n_topics} topics) in {time.time() - t0:.2f}s -> daily {daily_mat.shape}")
    return daily_mat, pd.DataFrame(topic_rows)


def _cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu <= 1e-9 or nv <= 1e-9:
        return 0.0
    cos = float(np.dot(u, v) / (nu * nv))
    return float(np.clip(1.0 - cos, 0.0, 2.0))


def build_macro_domain_features(
    daily_finbert_768: np.ndarray,
    merged_news: pd.DataFrame,
    article_emb_map: dict,
    sent_prob_map: dict,
    all_trading_dates: np.ndarray,
):
    """
    Strictly causal Macro Domain Features (no future leakage):
      1. novelty_5d: 1 - cos(e_t, mean(e_{t-5:t-1}))
      2. novelty_20d: 1 - cos(e_t, mean(e_{t-20:t-1}))
      3. emb_dispersion: mean cosine distance of day t articles vs daily centroid e_t
      4. finbert_net_sent: mean(P(pos) - P(neg)) of day t headlines
      5. finbert_sent_disagreement: std(P(pos) - P(neg)) across day t headlines
      6. finbert_neg_ratio: share of headlines with argmax == negative
      7. finbert_sent_surprise_5d: finbert_net_sent_t - mean(finbert_net_sent_{t-5:t-1})
      8. volume_shock_z20: causal z-score of daily article count vs prior 20 days
      9. log_article_count: log1p(n_articles)
    """
    n_days = len(all_trading_dates)
    aids = merged_news["article_id"].astype(str).tolist()
    dates = merged_news["effective_trading_date"].values

    sent_arr = np.vstack([sent_prob_map[aid] for aid in aids]).astype(np.float32)  # [pos, neg, neu]
    net_sent = sent_arr[:, 0] - sent_arr[:, 1]
    is_neg = (np.argmax(sent_arr, axis=1) == 1).astype(np.float32)
    embs = np.vstack([article_emb_map[aid] for aid in aids]).astype(np.float32)

    date_to_indices = {}
    for idx, d in enumerate(dates):
        date_to_indices.setdefault(d, []).append(idx)

    emb_dispersion = np.zeros(n_days, dtype=np.float32)
    sent_mean = np.zeros(n_days, dtype=np.float32)
    sent_std = np.zeros(n_days, dtype=np.float32)
    neg_ratio = np.zeros(n_days, dtype=np.float32)
    art_counts = np.zeros(n_days, dtype=np.float32)

    for t, d in enumerate(all_trading_dates):
        idxs = date_to_indices.get(d, [])
        art_counts[t] = float(len(idxs))
        if not idxs:
            continue
        idxs_arr = np.array(idxs, dtype=int)
        sent_mean[t] = float(np.mean(net_sent[idxs_arr]))
        sent_std[t] = float(np.std(net_sent[idxs_arr])) if len(idxs_arr) > 1 else 0.0
        neg_ratio[t] = float(np.mean(is_neg[idxs_arr]))

        day_embs = embs[idxs_arr]
        centroid = daily_finbert_768[t]
        if len(idxs_arr) > 1:
            emb_dispersion[t] = float(np.mean([_cosine_distance(e, centroid) for e in day_embs]))
        else:
            emb_dispersion[t] = 0.0

    novelty_5d = np.zeros(n_days, dtype=np.float32)
    novelty_20d = np.zeros(n_days, dtype=np.float32)
    sent_surprise_5d = np.zeros(n_days, dtype=np.float32)
    vol_shock_z20 = np.zeros(n_days, dtype=np.float32)

    for t in range(1, n_days):
        w5_start = max(0, t - 5)
        w20_start = max(0, t - 20)

        prev_mean_5 = daily_finbert_768[w5_start:t].mean(axis=0)
        prev_mean_20 = daily_finbert_768[w20_start:t].mean(axis=0)
        novelty_5d[t] = _cosine_distance(daily_finbert_768[t], prev_mean_5)
        novelty_20d[t] = _cosine_distance(daily_finbert_768[t], prev_mean_20)

        sent_surprise_5d[t] = sent_mean[t] - float(np.mean(sent_mean[w5_start:t]))

        prev_counts = art_counts[w20_start:t]
        if len(prev_counts) >= 3:
            mu = float(np.mean(prev_counts))
            sd = float(np.std(prev_counts))
            vol_shock_z20[t] = float((art_counts[t] - mu) / (sd + 1e-6))

    log_count = np.log1p(art_counts).astype(np.float32)

    feature_names = [
        "novelty_5d",
        "novelty_20d",
        "emb_dispersion",
        "finbert_net_sent",
        "finbert_sent_disagreement",
        "finbert_neg_ratio",
        "finbert_sent_surprise_5d",
        "volume_shock_z20",
        "log_article_count",
    ]
    macro_mat = np.column_stack([
        novelty_5d,
        novelty_20d,
        emb_dispersion,
        sent_mean,
        sent_std,
        neg_ratio,
        sent_surprise_5d,
        vol_shock_z20,
        log_count,
    ]).astype(np.float32)

    return macro_mat, feature_names
