Kelompok Ayam Bakar Pak Yanto:
Aloysius Pijar Hutama Indrianto 24/534591/PA/22675
Danar Fathurahman 24/5348200/PA/22828
Gilbert Nathaniel 24/533877/PA/22623
Satya Wira Pramudita 24/543649/PA/23102

# Pipeline Tugas 1 NLP - Berita Geopolitik & Nilai Tukar USD/IDR

## Struktur repository
- `data/raw/` - `news_raw_sample.csv`, `exchange_rate_bi_raw.csv` (Kurs.csv JISDOR dari BI)
- `data/cleaned/` - `news_cleaned.csv` (14.941 artikel geopolitik; file besar lokal/Drive), `news_cleaned_sample.csv`, `exchange_rate_bi_cleaned.csv`
- `data/aligned/` - `news_alignment.csv`, `daily_aligned_dataset.csv`
- `data/metadata/` - `filtering_log.csv`, `removed_articles.csv`, `data_dictionary.csv`
- `notebook/pipeline_tugas2.ipynb` - notebook final mandiri (audit/EDA sampai evaluasi)

## Cara menjalankan
- Notebook final mandiri: `notebook/pipeline_tugas2.ipynb` (seluruh proses inline; aman Run All).
- Pipeline scraping Tugas 1 (sitemap CNBC + 4 shard DB) diarsipkan di luar repository (Drive); hasil akhirnya tersedia di `data/`.

## Aturan utama (ringkas)
- Sumber berita: CNBC (1 Sep 2021 - 1 Sep 2026). Kurs: JISDOR Bank Indonesia.
- Filter geopolitik: kategori inti (Tier A) atau keyword judul + kategori politik/world; hanya artikel berkonten.
- Cleaning: normalisasi unicode + buang boilerplate; dedup konten dalam grup judul sama (rasio >= 0,99); buang artikel tanpa judul dan tanpa hari perdagangan berikutnya.
- Alignment: setiap berita dipetakan ke hari perdagangan PERTAMA setelah tanggal publikasi WIB (kurs terbit tengah malam, tidak ada look-ahead).
- Hasil: 14.941 artikel dipetakan ke 1.202 hari perdagangan; `daily_aligned_dataset.csv` siap untuk tugas berikutnya.
## Catatan
- Jangan commit file besar (`*.db`, `cnbc_articles*.csv`, `data/cleaned/news_cleaned.csv`); gunakan Drive. Sampel kecil tersedia di `*_sample.csv`.

# Pipeline Tugas 2 NLP - Eksperimen Baseline Arah USD/IDR

Tujuan: menguji apakah fitur NLP dari berita geopolitik menambah sinyal prediksi arah kurs USD/IDR hari berikutnya di luar fitur kurs historis.

## Struktur tambahan
- `src/config.py` - konfigurasi eksperimen (seed, periode split, daftar fitur, hyperparameter)
- `src/data_preparation.py` - audit dataset, target arah, split kronologis, verifikasi leakage
- `src/feature_market.py` - fitur pasar (lag return, MA, volatilitas; jendela berakhir di t)
- `src/feature_sentiment.py` - skor VADER + Loughran-McDonald dan agregasi harian
- `src/feature_tfidf.py` - dokumen harian + TF-IDF/SVD (fit hanya di train)
- `src/baseline.py` - baseline naif (all-up, persistence)
- `src/train.py` - pemilihan text source TF-IDF di validation + pelatihan model0-model3
- `src/evaluate.py` - metrik, tabel hasil, figure, diagram pipeline
- `src/make_report_pdf.py` - regenerate `reports/laporan_tugas2.pdf` dari `laporan_tugas2.md` (Chrome headless)
- `notebook/pipeline_tugas2.ipynb` - notebook final mandiri: EDA/audit sampai evaluasi, seluruh proses inline dengan penjelasan tiap langkah
- `data/processed/` - `train.csv`, `validation.csv`, `test.csv`, `audit_summary.json`
- `reports/` - laporan Tugas 2 (`laporan_tugas2.md` + `laporan_tugas2.pdf`), `results_*.csv`, `predictions_*.csv`, `figures/` (lokal; tidak di-commit)

## Cara menjalankan
1. `pip install vaderSentiment pysentiment2 matplotlib scikit-learn pandas markdown`
2. `python -m src.data_preparation` (skoring sentimen 14.941 artikel di-cache ke `data/cleaned/news_sentiment.csv`)
3. `python -m src.train`
4. `python -m src.evaluate`
5. `python src/make_report_pdf.py` (membuat `reports/laporan_tugas2.pdf`; butuh Chrome/Edge)
6. Notebook final mandiri: `notebook/pipeline_tugas2.ipynb` (semua kode inline, menghasilkan output yang sama)

## Aturan utama (ringkas)
- Target: `target_next_up = 1` jika kurs hari perdagangan berikutnya lebih tinggi, flat = 0.
- Cutoff konservatif: fitur hari t hanya memakai kurs <= t dan berita terbit sebelum hari t (aturan next trading day Tugas 1).
- Split kronologis tanpa shuffle: train s/d 2024-12, validation 2025, test 2026; test dievaluasi sekali.
- `StandardScaler`, `TfidfVectorizer`, dan `TruncatedSVD` selalu di-fit di dalam pipeline pada data train.
- Hasil awal: fitur NLP belum terbukti menambah sinyal stabil (test AUC model0 0,567 vs model3 0,568; baseline selalu-naik akurasi 0,571). Detail di `reports/laporan_tugas2.md`.
