Kelompok Ayam Bakar Pak Yanto:
Aloysius Pijar Hutama Indrianto 24/534591/PA/22675
Danar Fathurahman 24/5348200/PA/22828
Gilbert Nathaniel 24/533877/PA/22623
Satya Wira Pramudita 24/543649/PA/23102

# Pipeline Tugas 1 NLP - Berita Geopolitik & Nilai Tukar USD/IDR

## Struktur repository
- `scraper_v1/` - notebook pipeline end-to-end (`FINALSCRAPING2.ipynb`), sitemap mentah, DB hasil scraping 4 shard
- `src/pipeline_tugas1.ipynb` - salinan final notebook pipeline (deliverable kode)
- `reports/` - laporan Tugas 1 (`laporan_tugas1.md`) + `figures/` (workflow Mermaid + chart)
- `data/raw/` - `news_raw_sample.csv`, `exchange_rate_bi_raw.csv` (Kurs.csv JISDOR dari BI)
- `data/cleaned/` - `news_cleaned.csv` (14.941 artikel geopolitik), `news_cleaned_sample.csv`, `exchange_rate_bi_cleaned.csv`
- `data/aligned/` - `news_alignment.csv`, `daily_aligned_dataset.csv`
- `data/metadata/` - `filtering_log.csv`, `removed_articles.csv`, `data_dictionary.csv`

## Cara menjalankan
1. Buka `scraper_v1/FINALSCRAPING2.ipynb` (butuh `merged_sitemaps.csv`, DB hasil scraping, dan `data/raw/exchange_rate_bi_raw.csv`).
2. `RUN_SCRAPING = False` dan `RUN_MERGE = False` (default): cell scraping/merge di-skip, aman Run All di mesin mana pun; hasil preprocessing + alignment ditulis ke folder `data/` di root repo.
3. Set `RUN_SCRAPING = True` hanya di mesin yang punya `urls_shard_*.csv` + DB shard; set `RUN_MERGE = True` untuk menggabungkan 4 shard DB.

## Aturan utama (ringkas)
- Sumber berita: CNBC (1 Sep 2021 - 1 Sep 2026). Kurs: JISDOR Bank Indonesia.
- Filter geopolitik: kategori inti (Tier A) atau keyword judul + kategori politik/world; hanya artikel berkonten.
- Cleaning: normalisasi unicode + buang boilerplate; dedup konten dalam grup judul sama (rasio >= 0,99); buang artikel tanpa judul dan tanpa hari perdagangan berikutnya.
- Alignment: setiap berita dipetakan ke hari perdagangan PERTAMA setelah tanggal publikasi WIB (kurs terbit tengah malam, tidak ada look-ahead).
- Hasil: 14.941 artikel dipetakan ke 1.202 hari perdagangan; `daily_aligned_dataset.csv` siap untuk tugas berikutnya.
## Catatan
- Jangan commit file besar (`*.db`, `cnbc_articles*.csv`, `data/cleaned/news_cleaned.csv`); gunakan Drive. Sampel kecil tersedia di `*_sample.csv`.
