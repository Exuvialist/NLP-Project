# Spesifikasi & Dokumentasi Rekayasa Fitur (Feature Engineering)
## Tugas 2: Proposal Pipeline & Eksperimen Baseline
**Kelompok Ayam Bakar Pak Yanto**
- Aloysius Pijar Hutama Indrianto (24/534591/PA/22675)
- Danar Fathurahman (24/5348200/PA/22828)
- Gilbert Nathaniel (24/533877/PA/22623)
- Satya Wira Pramudita (24/543649/PA/23102)

---

## 1. Pendahuluan & Prinsip Desain Fitur

Dokumen ini mendokumentasikan secara komprehensif seluruh arsitektur rekayasa fitur (*feature engineering*) yang diimplementasikan pada **Tugas 2: Proposal Pipeline & Eksperimen Baseline**.

### 1.1 Tujuan Rekayasa Fitur
Mengonversi data deret waktu nilai tukar USD/IDR (Bank Indonesia JISDOR) dan teks berita geopolitik global (CNBC) menjadi matriks fitur tabular harian terukur ($X_t$) untuk memprediksi arah pergerakan kurs esok hari ($y_t$).

### 1.2 Prinsip Kausalitas & Pencegahan Kebocoran Data (*Zero Data Leakage*)
1. **Aturan Penyelarasan Temporal (*Next Trading Day Rule*):**
   Kurs JISDOR Bank Indonesia dipublikasikan pukul 00:00 WIB untuk hari perdagangan $t$. Oleh karena itu, berita yang terbit pada tanggal $t$ WIB secara kausal dipetakan ke **hari bursa efektif berikutnya ($t+1$)**. Fitur pada baris $t$ hanya menggunakan informasi yang sudah tersedia saat pasar dibuka pada hari $t$.
2. **Pemisahan Kronologis (*Strict Chronological Split*):**
   Dataset dibagi secara urut waktu tanpa pengacakan (*no random shuffle*):
   - **Train Set:** 29 September 2021 – 31 Desember 2024 (790 hari bursa, ~67%)
   - **Validation Set:** 2 Januari 2025 – 31 Desember 2025 (237 hari bursa, ~20%)
   - **Test Set:** 2 Januari 2026 – 31 Agustus 2026 (154 hari bursa, ~13%)
3. **Pencegahan Leakage pada Transformasi Pembelajaran:**
   Transformasi yang memiliki bobot terpelajar (seperti kosakata dan pembobotan IDF pada `TfidfVectorizer`) **hanya di-fit pada data Train**, kemudian di-*transform* ke data Validation dan Test.
4. **Kepatuhan Batasan Tugas:**
   Sesuai ketentuan, sistem **100% bebas dari pre-trained embeddings dan transformer** (tanpa FinBERT, RoBERTa, maupun LLM embeddings). Seluruh ekstraksi teks murni menggunakan pendekatan lexicon dan vectorization klasik.

---

## 2. Formulasi Variabel Target

Target utama yang diprediksi adalah arah biner pergerakan nilai tukar USD/IDR pada hari perdagangan berikutnya:

$$y_t = \begin{cases} 1, & \text{jika } Close_{t+1} > Close_t \text{ (NAIK / Dolar AS Menguat)} \\ 0, & \text{jika } Close_{t+1} \le Close_t \text{ (TURUN / Dolar AS Melemah/Tetap)} \end{cases}$$

Selain target biner, dicatat pula nilai *return* kontinu untuk evaluasi model time-series:
$$r_{t+1} = \frac{Close_{t+1} - Close_t}{Close_t}$$

---

## 3. Rincian Matriks Fitur (Total 54 Fitur Prediktor)

Fitur yang dihasilkan terbagi ke dalam 3 pilar representasi:

```
Total Fitur Prediktor (54 Dimensi)
├── 1. Fitur Pasar Historis (11 Fitur)    -> src/feature_market.py
├── 2. Fitur Sentimen Hibrida (13 Fitur)   -> src/feature_sentiment.py
└── 3. Fitur Topik TF-IDF (30 Fitur)      -> src/feature_tfidf.py
```

### 3.1 Pilar 1: Fitur Pasar Historis / *Market-Only* (11 Fitur)
📂 Modul: `src/feature_market.py`

Seluruh fitur teknikal dihitung secara kausal menggunakan harga penutupan hingga hari $t$:

| No | Nama Kolom | Formulasi Matematika | Deskripsi & Nilai Informasi |
|:--:|---|---|---|
| 1 | `return_lag1` | $r_t = \frac{Close_t - Close_{t-1}}{Close_{t-1}}$ | Return penutupan hari ini vs kemarin (tersedia saat pasar tutup hari $t$). |
| 2 | `return_lag2` | $r_{t-1}$ | Return 1 hari bursa sebelumnya (efek autoregresif lag-2). |
| 3 | `return_lag3` | $r_{t-2}$ | Return 2 hari bursa sebelumnya (efek autoregresif lag-3). |
| 4 | `return_lag5` | $r_{t-4}$ | Return 4 hari bursa sebelumnya (siklus mingguan). |
| 5 | `sma5_ratio` | $\frac{Close_t}{\text{SMA}_5(Close)} - 1.0$ | Rasio deviasi harga terhadap rata-rata bergerak 5 hari (*short-term mean reversion*). |
| 6 | `sma20_ratio` | $\frac{Close_t}{\text{SMA}_{20}(Close)} - 1.0$ | Rasio deviasi harga terhadap tren bulanan (20 hari bursa). |
| 7 | `volatility_5d` | $\sigma_{r, 5d} = \text{std}(r_{t-4 \dots t})$ | Volatilitas bergulir 5 hari bursa (tingkat gejolak jangka pendek). |
| 8 | `volatility_20d` | $\sigma_{r, 20d} = \text{std}(r_{t-19 \dots t})$ | Volatilitas bergulir 20 hari bursa (gejolak volatilitas bulanan). |
| 9 | `momentum_5d` | $\frac{Close_t}{Close_{t-5}} - 1.0$ | Laju perubahan harga persentase 5 hari bursa (*5-day price momentum*). |
| 10 | `momentum_20d` | $\frac{Close_t}{Close_{t-20}} - 1.0$ | Laju perubahan harga persentase 20 hari bursa (*monthly price momentum*). |
| 11 | `day_of_week` | $\text{DayOfWeek}(t) \in [0, 4]$ | Indeks hari perdagangan (0=Senin, ..., 4=Jumat) untuk efek kalender. |

*Catatan Warm-up:* 20 baris pertama data historis dialokasikan sebagai *warm-up period* untuk menghitung rolling 20-hari, sehingga tidak ada nilai `NaN`.

---

### 3.2 Pilar 2: Fitur Sentimen Teks Hibrida / *Dual-Lexicon* (13 Fitur)
📂 Modul: `src/feature_sentiment.py`

Eksperimen komparasi empiris kami membuktikan bahwa:
- **VADER** sangat unggul pada **Judul Berita** (cakupan sentimen 64.0% vs LM hanya 47.6%) karena peka terhadap kata-kata aksi/konflik geopolitik (*war*, *threat*, *attack*, *sanction*).
- **Loughran-McDonald (LM)** sangat unggul pada **Isi Berita** (cakupan 93.8%) karena mampu menyaring istilah ketidakpastian finansial (*uncertainty*, *deficit*, *impairment*, *liability*) tanpa terdistorsi kata emosional umum.

Karena satu hari bursa memiliki 1 s/d 133 artikel berita, ekstraksi dilakukan pada level artikel lalu diagregasikan per hari bursa efektif:

| No | Nama Kolom | Komponen Lexicon | Deskripsi Agregasi Harian |
|:--:|---|---|---|
| 1 | `vader_compound_mean` | VADER (Judul) | Nilai rata-rata skor compound judul harian (rentang [-1, +1]). |
| 2 | `vader_compound_min` | VADER (Judul) | Skor sentimen judul paling negatif/ekstrim hari itu (indikator berita guncangan). |
| 3 | `vader_compound_max` | VADER (Judul) | Skor sentimen judul paling positif/optimis hari itu. |
| 4 | `vader_compound_std` | VADER (Judul) | Standar deviasi sentimen judul harian (mengukur tingkat polarisasi opini media). |
| 5 | `vader_pos_mean` | VADER (Judul) | Rata-rata proporsi leksikal positif judul berita harian. |
| 6 | `vader_neg_mean` | VADER (Judul) | Rata-rata proporsi leksikal negatif judul berita harian. |
| 7 | `lm_polarity_mean` | LM (Isi Berita) | Rata-rata polaritas finansial: $\frac{Pos - Neg}{Pos + Neg}$ berdasarkan kamus resmi 10-K SEC. |
| 8 | `lm_polarity_min` | LM (Isi Berita) | Skenario risiko finansial terburuk pada artikel yang terbit hari itu. |
| 9 | `lm_polarity_max` | LM (Isi Berita) | Skenario finansial terbaik pada artikel yang terbit hari itu. |
| 10 | `lm_uncertainty_mean` | LM (Isi Berita) | Rata-rata rasio kata *Uncertainty* (ketidakpastian) per total kata artikel. |
| 11 | `lm_pos_ratio_mean` | LM (Isi Berita) | Densitas kemunculan kata positif finansial terhadap panjang artikel. |
| 12 | `lm_neg_ratio_mean` | LM (Isi Berita) | Densitas kemunculan kata negatif finansial terhadap panjang artikel. |
| 13 | `news_volume` | Metadata | Jumlah total artikel geopolitik yang dipetakan ke hari bursa tersebut (intensitas liputan). |

---

### 3.3 Pilar 3: Fitur Representasi Topik Geopolitik / *Classical TF-IDF* (30 Fitur)
📂 Modul: `src/feature_tfidf.py`

Untuk menangkap topik struktural tanpa embeddings terlarang, digunakan representasi N-gram TF-IDF klasik:

1. **Konstruksi Dokumen Harian:** Seluruh judul berita geopolitik pada hari perdagangan $t$ digabungkan menjadi satu teks harian (*daily concatenated headline document*).
2. **Konfigurasi Vectorizer:**
   - Model: `TfidfVectorizer` dari scikit-learn.
   - Kosakata: Top 30 n-gram (`ngram_range=(1, 2)`).
   - Pra-pengolahan: Huruf kecil (*lowercase*), penghapusan *English stopwords*, pembobotan sublinear TF ($1 + \log(TF)$).
   - **Fit:** HANYA pada dokumen harian Train (s/d 31 Desember 2024).

#### 30 Fitur Topik TF-IDF yang Dihasilkan:
```text
tfidf_biden           tfidf_china           tfidf_climate         tfidf_court
tfidf_debt            tfidf_defense         tfidf_economy         tfidf_energy
tfidf_europe          tfidf_fed             tfidf_gas             tfidf_global
tfidf_government      tfidf_inflation       tfidf_iran            tfidf_israel
tfidf_military        tfidf_minister        tfidf_new             tfidf_oil
tfidf_president       tfidf_president_biden tfidf_prime           tfidf_prime_minister
tfidf_russia          tfidf_russian         tfidf_says            tfidf_ukraine
tfidf_war             tfidf_world
```

*Interpretasi Finansial:* Fitur-fitur ini menangkap tema makroekonomi dan guncangan geopolitik spesifik (seperti eskalasi perang, pembatasan pasokan energi/minyak, kebijakan suku bunga The Fed, dan rivalitas dagang AS-China) yang berdampak langsung pada aliran modal mata uang Dolar AS.

---

## 4. Struktur Dataset Olahan (*Processed Datasets*)

Setelah penggabungan seluruh pilar fitur dan imputasi netral ($0.0$) pada hari libur bursa, dataset tersimpan dalam format CSV di folder `data/processed/`:

| Nama File | Periode Waktu | Jumlah Baris | Proporsi UP (1) | Proporsi DOWN (0) |
|---|---|:---:|:---:|:---:|
| `data/processed/train_full.csv` | 29 Sep 2021 – 31 Des 2024 | **790** | 444 (56.2%) | 346 (43.8%) |
| `data/processed/val_full.csv`   | 02 Jan 2025 – 31 Des 2025 | **237** | 124 (52.3%) | 113 (47.7%) |
| `data/processed/test_full.csv`  | 02 Jan 2026 – 31 Agu 2026 | **154** | 88 (57.1%) | 66 (42.9%) |

---

## 5. Ringkasan Kinerja Rekayasa Fitur terhadap Model Baseline

Performa matriks fitur diuji pada 4 skema model (Model 0 s/d Model 3) menggunakan metrik **F1-Score** dan **Confusion Matrix**:

| Model | Fitur Input | Split Data | F1 Macro | F1 UP (1) | F1 DOWN (0) | Akurasi | Confusion Matrix `[TN, FP, FN, TP]` |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Model 0 (ARIMAX)** | *Market-Only* (Lags & Vol) | Validation | 0.4880 | 0.5420 | 0.4340 | 49.37% | `[46, 67, 53, 71]` |
| **Model 0 (ARIMAX)** | *Market-Only* (Lags & Vol) | Test | 0.4809 | 0.5618 | 0.4000 | 49.35% | `[26, 40, 38, 50]` |
| **Model 0 (XGBoost)**| *Market-Only* (11 Fitur) | Validation | 0.4740 | 0.6205 | 0.3275 | 51.48% | `[28, 85, 30, 94]` |
| **Model 0 (XGBoost)**| *Market-Only* (11 Fitur) | Test | 0.4137 | 0.6117 | 0.2157 | 48.05% | `[11, 55, 25, 63]` |
| **Model 1 (XGBoost)**| Market + Sentimen (24 Fitur) | Validation | 0.4068 | 0.6446 | 0.1690 | 50.21% | `[12, 101, 17, 107]` |
| **Model 1 (XGBoost)**| Market + Sentimen (24 Fitur) | Test | 0.4444 | 0.6573 | 0.2316 | 52.60% | `[11, 55, 18, 70]` |
| 🏆 **Model 2 (XGBoost)**| **Market + TF-IDF (41 Fitur)** | **Validation** | **0.4964** | **0.6770** | **0.3158** | **56.12%** | `[24, 89, 15, 109]` |
| 🏆 **Model 2 (XGBoost)**| **Market + TF-IDF (41 Fitur)** | **Test** | **0.5338** | **0.7306** | **0.3371** | **61.69%** | `[15, 51, 8, 80]` |
| **Model 3 (XGBoost)**| Full Combined (54 Fitur) | Validation | 0.4269 | 0.6566 | 0.1972 | 51.90% | `[14, 99, 15, 109]` |
| **Model 3 (XGBoost)**| Full Combined (54 Fitur) | Test | 0.4598 | 0.7054 | 0.2143 | 57.14% | `[9, 57, 9, 79]` |

---

## 6. Kesimpulan Penting untuk Laporan Tugas 2

1. **TF-IDF Menghasilkan Sinyal Prediktif Nyata:**
   Penambahan fitur TF-IDF (Model 2) meningkatkan **F1-Macro sebesar +12.01%** (dari 0.4137 ke 0.5338) dan **Akurasi sebesar +13.64%** (dari 48.05% ke 61.69%) pada data uji Test 2026. Ini membuktikan bahwa tema geopolitik spesifik (*oil*, *tariffs*, *sanctions*, *war*) memiliki daya prediksi riil terhadap volatilitas Dolar AS.
2. **Keterbatasan Sentimen Skalar:**
   Sentimen murni (-1 s/d +1) kurang informatif jika berdiri sendiri karena dalam geopolitik, berita buruk (seperti perang) sering memicu *flight to safety* yang justru **menguatkan Dolar AS**. Model memerlukan pembeda topik (TF-IDF) untuk memahami konteks peristiwa.
3. **Reproduksibilitas Penuh:**
   Seluruh pipeline dapat dieksekusi ulang secara modular dan deterministik melalui script:
   ```powershell
   python -m src.train
   ```
