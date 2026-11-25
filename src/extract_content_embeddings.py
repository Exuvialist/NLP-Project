import os
import sys
import time
import shutil
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

PROJECT_ROOT = Path("NLP-Project")
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
ALIGNED_DIR = DATA_DIR / "aligned"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = PROCESSED_DIR / "embeddings_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Extracting on device: {device}")
if device.type == 'cuda':
    print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

# 1. Load data
news_df = pd.read_csv(CLEANED_DIR / 'news_cleaned.csv', usecols=['article_id', 'title_clean', 'content_clean'])
align_df = pd.read_csv(ALIGNED_DIR / 'news_alignment.csv', usecols=['article_id', 'effective_trading_date'])
align_df['effective_trading_date'] = pd.to_datetime(align_df['effective_trading_date']).dt.strftime('%Y-%m-%d')

merged_news = align_df.merge(news_df, on='article_id', how='inner')
merged_news['title_clean'] = merged_news['title_clean'].fillna('')
merged_news['content_clean'] = merged_news['content_clean'].fillna('')
merged_news['title_content'] = merged_news['title_clean'] + ". " + merged_news['content_clean']

# Get all trading dates
full_market_df = pd.concat([
    pd.read_csv(PROCESSED_DIR / 'train.csv', parse_dates=['trading_date']),
    pd.read_csv(PROCESSED_DIR / 'validation.csv', parse_dates=['trading_date']),
    pd.read_csv(PROCESSED_DIR / 'test.csv', parse_dates=['trading_date'])
], ignore_index=True).sort_values('trading_date').reset_index(drop=True)
all_trading_dates = full_market_df['trading_date'].dt.strftime('%Y-%m-%d').values
print(f"[*] Total trading dates: {len(all_trading_dates)}")
print(f"[*] Total articles: {len(merged_news)}")

# 2. Check title cache
title_cache_orig = CACHE_DIR / "finbert_prosus_daily.npy"
title_cache_new = CACHE_DIR / "finbert_title_daily.npy"
if title_cache_orig.exists() and not title_cache_new.exists():
    shutil.copy(title_cache_orig, title_cache_new)
    print(f"[*] Copied title cache to {title_cache_new}")

# 3. Model & Tokenizer
model_id = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()

def extract_embeddings_for_text_column(texts, max_length=256, batch_size=64):
    all_embeddings = []
    total = len(texts)
    t0 = time.time()
    
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(**encoded)
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                mean_pooled = (sum_embeddings / sum_mask).float().cpu().numpy()
                
            all_embeddings.append(mean_pooled)
            
            if (i // batch_size) % 50 == 0 or i + batch_size >= total:
                elapsed = time.time() - t0
                pct = min(100.0, (i + batch_size) / total * 100)
                print(f"    Batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} ({pct:.1f}%) - {elapsed:.1f}s")
                
    return np.vstack(all_embeddings)

# 4. Extract content_clean
content_cache = CACHE_DIR / "finbert_content_daily.npy"
if content_cache.exists():
    print(f"[*] Content cache already exists: {content_cache}")
else:
    print(f"[*] Extracting FinBERT for content_clean (max_length=256)...")
    content_embs = extract_embeddings_for_text_column(merged_news['content_clean'].tolist(), max_length=256, batch_size=64)
    merged_news['content_vec'] = list(content_embs)
    daily_content_map = merged_news.groupby('effective_trading_date')['content_vec'].apply(
        lambda vecs: np.mean(np.vstack(vecs), axis=0)
    ).to_dict()
    dim = content_embs.shape[1]
    daily_content = np.array([daily_content_map.get(d, np.zeros(dim)) for d in all_trading_dates])
    np.save(content_cache, daily_content)
    print(f"[+] Saved content embeddings: {content_cache} (shape: {daily_content.shape})")

# 5. Extract title_content
tc_cache = CACHE_DIR / "finbert_title_content_daily.npy"
if tc_cache.exists():
    print(f"[*] Title+Content cache already exists: {tc_cache}")
else:
    print(f"[*] Extracting FinBERT for title_content (max_length=256)...")
    tc_embs = extract_embeddings_for_text_column(merged_news['title_content'].tolist(), max_length=256, batch_size=64)
    merged_news['tc_vec'] = list(tc_embs)
    daily_tc_map = merged_news.groupby('effective_trading_date')['tc_vec'].apply(
        lambda vecs: np.mean(np.vstack(vecs), axis=0)
    ).to_dict()
    dim = tc_embs.shape[1]
    daily_tc = np.array([daily_tc_map.get(d, np.zeros(dim)) for d in all_trading_dates])
    np.save(tc_cache, daily_tc)
    print(f"[+] Saved title+content embeddings: {tc_cache} (shape: {daily_tc.shape})")

print("[*] All FinBERT text granularity embeddings successfully extracted and cached!")
