"""
nlp_ablation_features.py
========================
Leakage-free feature engineering helpers for the 4-stage NLP Ablation Pipeline:
1. Article-level ProsusAI/finbert headline embeddings (768d)
2. Topic Modeling: LDA vs BERTopic(FinBERT) fit STRICTLY on Train split
3. Named Entity Recognition (spaCy en_core_web_sm): Structural counts + Train-fit Entity TF-IDF SVD
"""

import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import LatentDirichletAllocation, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def extract_headline_finbert_article_embeddings(
    df_news: pd.DataFrame,
    cache_path: Path,
    batch_size: int = 128,
    max_length: int = 96,
) -> dict:
    """
    Returns dict mapping article_id -> 768d ProsusAI/finbert embedding of title_clean.
    Caches matrix and article_ids aligned to df_news order.
    """
    cache_path = Path(cache_path)
    ids_path = cache_path.with_name(cache_path.stem + "_ids.npy")

    df_unique = df_news.drop_duplicates("article_id").reset_index(drop=True)
    article_ids = df_unique["article_id"].astype(str).tolist()

    if cache_path.exists() and ids_path.exists():
        cached_ids = np.load(ids_path, allow_pickle=True).tolist()
        if cached_ids == article_ids:
            embs = np.load(cache_path)
            print(f"[FinBERT-Articles] Loaded cached headline embeddings: {embs.shape}")
            return dict(zip(article_ids, embs))

    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FinBERT-Articles] Extracting 768d headline embeddings for {len(article_ids)} articles on {device}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert").to(device)
    model.eval()

    titles = df_unique["title_clean"].fillna("").astype(str).tolist()
    all_embs = []

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
                out = model(**enc)
                tok_emb = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
                pooled = torch.sum(tok_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                all_embs.append(pooled.float().cpu().numpy())

    embs = np.vstack(all_embs)
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    np.save(cache_path, embs)
    np.save(ids_path, np.array(article_ids, dtype=object))
    print(f"[FinBERT-Articles] Done in {time.time() - t0:.2f}s -> {embs.shape}")
    return dict(zip(article_ids, embs))


def _clean_lexical_for_topic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = re.sub(r"<[^>]+>", " ", text.lower())
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def fit_transform_lda_daily(
    merged_news: pd.DataFrame,
    all_trading_dates: np.ndarray,
    train_mask: np.ndarray,
    n_topics: int = 15,
    random_state: int = 42,
):
    """
    Fits CountVectorizer + LDA strictly on Train headlines, then transforms all articles
    and aggregates daily topic distributions + topic entropy.
    """
    t0 = time.time()
    train_dates = set(all_trading_dates[train_mask])
    is_train = merged_news["effective_trading_date"].isin(train_dates).values

    texts = merged_news["title_clean"].fillna("").map(_clean_lexical_for_topic).tolist()
    train_texts = [txt for txt, flag in zip(texts, is_train) if flag]

    vec = CountVectorizer(stop_words="english", max_features=3000, min_df=3, ngram_range=(1, 2))
    X_train_bow = vec.fit_transform(train_texts)
    X_all_bow = vec.transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        max_iter=25,
        random_state=random_state,
        n_jobs=1,
    )
    lda.fit(X_train_bow)
    doc_topics = lda.transform(X_all_bow)  # (N_art, n_topics)

    entropy = -np.sum(doc_topics * np.log(np.clip(doc_topics, 1e-12, 1.0)), axis=1, keepdims=True)
    art_features = np.hstack([doc_topics, entropy])  # (N_art, n_topics + 1)

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

    # Top terms per topic
    vocab = np.array(vec.get_feature_names_out())
    topic_rows = []
    for tidx, comp in enumerate(lda.components_):
        top_ids = comp.argsort()[-10:][::-1]
        topic_rows.append({
            "Model": "LDA",
            "Topic_ID": tidx,
            "Top_Terms": ", ".join(vocab[top_ids]),
        })

    print(f"[LDA] Fit on {is_train.sum()} Train articles ({n_topics} topics) in {time.time() - t0:.2f}s -> daily {daily_mat.shape}")
    return daily_mat, pd.DataFrame(topic_rows)


def fit_transform_bertopic_finbert_daily(
    merged_news: pd.DataFrame,
    article_emb_map: dict,
    all_trading_dates: np.ndarray,
    train_mask: np.ndarray,
    n_out_dim: int = 16,
    random_state: int = 42,
):
    """
    Fits BERTopic using precomputed ProsusAI/finbert headline embeddings STRICTLY on Train articles,
    transforms Val/Test articles via .transform(), aggregates soft probabilities daily,
    and compresses to fixed n_out_dim via Train-only PCA (zero leakage).
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    t0 = time.time()
    train_dates = set(all_trading_dates[train_mask])
    is_train = merged_news["effective_trading_date"].isin(train_dates).values

    docs = merged_news["title_clean"].fillna("").astype(str).tolist()
    embs = np.vstack([article_emb_map[aid] for aid in merged_news["article_id"].astype(str)]).astype(np.float32)

    train_idx = np.where(is_train)[0]
    other_idx = np.where(~is_train)[0]

    train_docs = [docs[i] for i in train_idx]
    train_embs = embs[train_idx]
    other_docs = [docs[i] for i in other_idx]
    other_embs = embs[other_idx]

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
        n_jobs=1,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=45,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=1,
    )
    vectorizer_model = CountVectorizer(stop_words="english", min_df=3, ngram_range=(1, 2))

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=False,
    )

    train_topics, train_probs = topic_model.fit_transform(train_docs, embeddings=train_embs)
    other_topics, other_probs = topic_model.transform(other_docs, embeddings=other_embs)

    # Ensure 2D probability arrays
    if train_probs.ndim == 1:
        train_probs = train_probs.reshape(-1, 1)
    if other_probs.ndim == 1:
        other_probs = other_probs.reshape(-1, 1)

    n_art = len(docs)
    k_bt = train_probs.shape[1]
    all_probs = np.zeros((n_art, k_bt), dtype=np.float32)
    all_probs[train_idx] = train_probs
    all_probs[other_idx] = other_probs

    all_topics = np.zeros(n_art, dtype=int)
    all_topics[train_idx] = train_topics
    all_topics[other_idx] = other_topics

    outlier_prob = np.clip(1.0 - all_probs.sum(axis=1, keepdims=True), 0.0, 1.0)
    is_outlier = (all_topics == -1).astype(np.float32).reshape(-1, 1)
    art_topic_feats = np.hstack([all_probs, outlier_prob, is_outlier])

    feat_df = pd.DataFrame(art_topic_feats)
    feat_df["effective_trading_date"] = merged_news["effective_trading_date"].values
    daily_map = feat_df.groupby("effective_trading_date").mean().to_dict(orient="index")

    raw_dim = art_topic_feats.shape[1]
    daily_raw = np.array(
        [
            [daily_map[d][c] for c in range(raw_dim)] if d in daily_map else np.zeros(raw_dim)
            for d in all_trading_dates
        ],
        dtype=np.float32,
    )

    # Compress soft topic distributions to n_out_dim - 1 via Train-only PCA + keep daily outlier_rate
    probs_daily = daily_raw[:, :-1]
    outlier_rate_daily = daily_raw[:, -1:]

    n_comp = min(max(1, n_out_dim - 1), probs_daily.shape[1], int(train_mask.sum()) - 1)
    scaler = StandardScaler()
    pca = PCA(n_components=n_comp, random_state=random_state)
    scaled_tr = scaler.fit_transform(probs_daily[train_mask])
    pca.fit(scaled_tr)
    probs_reduced = pca.transform(scaler.transform(probs_daily))

    daily_mat = np.hstack([probs_reduced, outlier_rate_daily]).astype(np.float32)

    # Extract top terms
    topic_info = topic_model.get_topic_info()
    topic_rows = []
    for _, row in topic_info.head(16).iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        words = [w for w, _ in (topic_model.get_topic(tid) or [])[:10]]
        topic_rows.append({
            "Model": "BERTopic(FinBERT)",
            "Topic_ID": tid,
            "Count_Train": int(row["Count"]),
            "Top_Terms": ", ".join(words),
        })

    print(
        f"[BERTopic(FinBERT)] Discovered {k_bt} topics on Train in {time.time() - t0:.2f}s "
        f"-> compressed daily {daily_mat.shape}"
    )
    return daily_mat, pd.DataFrame(topic_rows)


def fit_transform_ner_daily(
    merged_news: pd.DataFrame,
    all_trading_dates: np.ndarray,
    train_mask: np.ndarray,
    n_svd: int = 16,
    random_state: int = 42,
):
    """
    Runs spaCy en_core_web_sm NER across all headlines, generating:
      - 8 structural NER density features per day
      - 16 TF-IDF SVD semantic entity features fit STRICTLY on Train split
    """
    import spacy

    t0 = time.time()
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "attribute_ruler"])
    titles = merged_news["title_clean"].fillna("").astype(str).tolist()

    struct_rows = []
    entity_docs = []
    label_counter = Counter()
    ent_text_counter = Counter()

    for title, doc in zip(titles, nlp.pipe(titles, batch_size=256)):
        counts = Counter(ent.label_ for ent in doc.ents)
        total_ents = len(doc.ents)
        n_words = max(1, len(title.split()))

        org_c = counts["ORG"]
        gpe_c = counts["GPE"]
        per_c = counts["PERSON"]
        norp_c = counts["NORP"]
        money_pct_c = counts["MONEY"] + counts["PERCENT"]
        geo_other_c = counts["LOC"] + counts["FAC"] + counts["EVENT"] + counts["LAW"]
        density = total_ents / n_words

        struct_rows.append([
            org_c,
            gpe_c,
            per_c,
            norp_c,
            money_pct_c,
            geo_other_c,
            total_ents,
            density,
        ])

        toks = []
        for ent in doc.ents:
            label_counter[ent.label_] += 1
            clean_txt = re.sub(r"\s+", "_", ent.text.strip().lower())
            clean_txt = re.sub(r"[^\w_]", "", clean_txt)
            if len(clean_txt) >= 2:
                tok = f"{ent.label_.lower()}__{clean_txt}"
                toks.append(tok)
                ent_text_counter[(ent.label_, ent.text.strip())] += 1
        entity_docs.append(" ".join(toks))

    # Aggregate 8 structural NER features daily (mean per article)
    struct_df = pd.DataFrame(struct_rows)
    struct_df["effective_trading_date"] = merged_news["effective_trading_date"].values
    struct_daily_map = struct_df.groupby("effective_trading_date").mean().to_dict(orient="index")
    daily_struct = np.array(
        [
            [struct_daily_map[d][c] for c in range(8)] if d in struct_daily_map else np.zeros(8)
            for d in all_trading_dates
        ],
        dtype=np.float32,
    )

    # Aggregate entity tokens per trading day -> Train-only TF-IDF + SVD
    ent_df = pd.DataFrame({
        "effective_trading_date": merged_news["effective_trading_date"].values,
        "ent_doc": entity_docs,
    })
    daily_ent_map = ent_df.groupby("effective_trading_date")["ent_doc"].apply(lambda s: " ".join(s)).to_dict()
    daily_ent_docs = [daily_ent_map.get(d, "") for d in all_trading_dates]
    train_ent_docs = [doc for doc, flag in zip(daily_ent_docs, train_mask) if flag]

    vec = TfidfVectorizer(max_features=3000, min_df=2, token_pattern=r"(?u)\b\S+\b")
    X_tr = vec.fit_transform(train_ent_docs)
    X_all = vec.transform(daily_ent_docs)

    svd = TruncatedSVD(n_components=min(n_svd, X_tr.shape[1] - 1), random_state=random_state)
    svd.fit(X_tr)
    daily_ent_svd = svd.transform(X_all).astype(np.float32)

    daily_ner_full = np.hstack([daily_struct, daily_ent_svd]).astype(np.float32)

    stats_rows = [
        {"Entity_Label": lbl, "Entity_Text": txt, "Count": cnt}
        for (lbl, txt), cnt in ent_text_counter.most_common(40)
    ]
    print(f"[NER spaCy] Processed {len(titles)} headlines in {time.time() - t0:.2f}s -> daily {daily_ner_full.shape}")
    return daily_ner_full, daily_struct, daily_ent_svd, pd.DataFrame(stats_rows), dict(label_counter)
