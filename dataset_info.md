# Informasi Dataset: Prediksi Multimodal Arah Kurs USD/IDR

Dokumen ini merangkum spesifikasi, struktur, dan protokol penyelarasan dataset multimodal yang digunakan pada proyek prediksi arah pergerakan kurs USD/IDR. Proyek ini memadukan data kuantitatif pasar keuangan (tabular) dengan data teks berita geopolitik dan ekonomi makro (NLP).

---

## 1. Dataset Tabular: Kurs Valuta Asing (JISDOR Bank Indonesia)

* **Sumber:** Bank Indonesia (JISDOR: *Jakarta Interbank Spot Dollar Rate*).
* **File Utama:** [exchange_rate_bi_cleaned.csv](file:///d:/Things%20For%20Kuliah/SEMESTER%205/NLP/NLP-Project/data/cleaned/exchange_rate_bi_cleaned.csv)
* **Rentang Waktu:** 1 September 2021 sampai 1 September 2026 (1.202 hari perdagangan).
* **Frekuensi:** Harian (hari kerja bursa Bank Indonesia).

### Fitur Teknikal Pasar (8 Dimensi)
Dari data deret waktu harga kurs, diekstraksi 8 fitur teknikal yang mencerminkan momentum, deviasi terhadap rata-rata, dan volatilitas:

| Nama Fitur | Definisi Matematis / Deskripsi | Interpretasi Pasar |
| :--- | :--- | :--- |
| `ret_1d` | $(P_t - P_{t-1}) / P_{t-1}$ | Return harian kurs kemarin |
| `ret_2d` | $(P_t - P_{t-2}) / P_{t-2}$ | Momentum return 2 hari perdagangan |
| `ret_3d` | $(P_t - P_{t-3}) / P_{t-3}$ | Momentum return 3 hari perdagangan |
| `ret_5d` | $(P_t - P_{t-5}) / P_{t-5}$ | Momentum mingguan (5 hari kerja bursa) |
| `rate_to_ma5` | $(P_t / \text{MA}_5) - 1$ | Deviasi kurs terhadap moving average 5 hari |
| `rate_to_ma20` | $(P_t / \text{MA}_{20}) - 1$ | Deviasi kurs terhadap moving average 20 hari |
| `vol_5` | $\sigma(\text{ret}_{t-4:t})$ | Volatilitas jangka pendek (standar deviasi return 5 hari) |
| `vol_20` | $\sigma(\text{ret}_{t-19:t})$ | Volatilitas bulanan (standar deviasi return 20 hari) |

### Variabel Target
* **Kolom:** `target_next_up`
* **Definisi:** $y_t = \mathbb{I}(P_{t+1} > P_t)$
  * `1` (Up): Kurs USD/IDR esok hari ditutup naik (nilai tukar Rupiah melemah terhadap Dolar AS).
  * `0` (Down/Flat): Kurs USD/IDR esok hari ditutup turun atau tidak berubah (Rupiah menguat atau stabil).

---

## 2. Dataset Teks: Berita Geopolitik & Ekonomi Makro

* **Sumber:** CNBC Indonesia.
* **File Utama:** [news_cleaned.csv](file:///d:/Things%20For%20Kuliah/SEMESTER%205/NLP/NLP-Project/data/cleaned/news_cleaned.csv)
* **Jumlah Artikel:** 14.941 artikel berita bersih (setelah deduplikasi dan pembersihan boilerplate).
* **Rentang Publikasi:** 1 September 2021 sampai 31 Agustus 2026.
* **Fokus Topik:** Geopolitik global, konflik militer, perang dagang/tarif, kebijakan suku bunga bank sentral (The Fed, BI), energi global, dan dinamika rantai pasok.

### Struktur Kolom Data Teks
* `article_id`: Pengenal unik artikel (format: `cnbc_<8-digit-md5>`).
* `title_clean`: Judul berita hasil normalisasi unicode, pembersihan tanda baca berlebih, dan penghapusan boilerplate.
* `content_clean`: Teks isi lengkap berita yang telah dibersihkan.
* `published_at_wib`: Waktu publikasi resmi dalam zona Waktu Indonesia Barat (WIB).
* `match_rule`: Kategori penyaringan artikel (`tier_a` untuk kategori inti geopolitik, atau kata kunci berbasis kebijakan makro).

---

## 3. Protokol Penyelarasan Waktu (*Temporal Alignment & Zero Leakage*)

* **File Penyelarasan:** [news_alignment.csv](file:///d:/Things%20For%20Kuliah/SEMESTER%205/NLP/NLP-Project/data/aligned/news_alignment.csv) dan [daily_aligned_dataset.csv](file:///d:/Things%20For%20Kuliah/SEMESTER%205/NLP/NLP-Project/data/aligned/daily_aligned_dataset.csv).
* **Aturan Penyelarasan (`next_trading_day`):**
  * Berita yang terbit pada akhir pekan (Sabtu-Minggu) atau hari libur nasional dipetakan ke hari perdagangan bursa berikutnya.
  * Berita yang terbit setelah batas waktu penutupan pasar JISDOR pada hari $t$ dipetakan ke hari perdagangan $t+1$.
  * Hal ini menjamin prinsip *Zero Look-Ahead Bias*: fitur teks dan teknikal pada hari $t$ hanya memuat informasi yang sudah tersedia untuk publik sebelum target $t+1$ terjadi.
* **Karakteristik Agregasi Harian:**
  * Rata-rata artikel per hari perdagangan: 12,43 artikel.
  * Median: 10 artikel per hari.
  * Maksimum: 133 artikel (saat terjadi eskalasi geopolitik besar).
  * Hari perdagangan dengan artikel: 1.201 dari 1.202 hari (99,92% coverage).
  * Seluruh teks artikel pada hari yang sama digabungkan (*concatenated daily text*) untuk diekstrak fiturnya menjadi representasi vektor teks berdimensi 32 (TF-IDF, GloVe, FastText, FinBERT, dsb.).

---

## 4. Pembagian Partisi Data (*Temporal Splits*)

Pemisahan data tidak dilakukan secara acak (random split) untuk menghindari kebocoran data deret waktu (*temporal leakage*). Pembagian berbasis kronologis kalender:

| Partisi | Rentang Tanggal | Jumlah Hari Trading | Rasio Relatif | Fungsi Utama |
| :--- | :--- | :--- | :--- | :--- |
| **Train Set** | $\le$ 31 Desember 2024 | 790 hari | ~67% | Pelatihan model, fitting SVD/PCA reduksi dimensi teks |
| **Validation Set** | 1 Januari 2025 s.d. 31 Desember 2025 | 237 hari | ~20% | Penyetelan hyperparameter, seleksi representasi embedding |
| **Test Set** | 1 Januari 2026 s.d. 1 September 2026 | 154 hari | ~13% | Evaluasi final out-of-sample pada data pasar terbaru |

---

## 5. Ringkasan File & Direktori Terkait

* [exchange_rate_bi_cleaned.csv](file:///d:/Things%20For%20Kuliah/SEMESTER%205/NLP/NLP-Project/data/cleaned/exchange_rate_bi_cleaned.csv): Data kurs bersih harian Bank Indonesia.
* [news_cleaned.csv](file:///d:/Things%20For%20Kuliah/SEMESTER%205/NLP/NLP-Project/data/cleaned/news_cleaned.csv): Korpus berita geopolitik dan makroekonomi bersih.
* [daily_aligned_dataset.csv](file:///d:/Things%20For%20Kuliah/SEMESTER%205/NLP/NLP-Project/data/aligned/daily_aligned_dataset.csv): Penggabungan harian data kurs dan referensi artikel berita.
* [audit_summary.json](file:///d:/Things%20For%20Kuliah/SEMESTER%205/NLP/NLP-Project/data/processed/audit_summary.json): Laporan audit statistik dan integritas metadata dataset.
* [data_dictionary.csv](file:///d:/Things%20For%20Kuliah/SEMESTER%205/NLP/NLP-Project/data/metadata/data_dictionary.csv): Kamus definisi setiap kolom dalam dataset proyek.
