"""
plot_experiment_pipeline_diagram.py
==================================
Generates a clean, publication-grade IEEE 2-column horizontal flowchart diagram
showing the end-to-end 5-Phase experimental methodology (Bab II Metodologi),
from Data Acquisition and Market Feature Engineering through Embeddings,
Granularity/Pre-Filtering, Complementary Features, Validation Gate, Fusion Strategies,
and Final Evaluation (without locking into final winning feature choices).
"""

from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon


def draw_rounded_box(ax, x, y, w, h, title, body_lines, header_color, face_color="#FFFFFF", border_color="#2C3E50", lw=1.4):
    # Outer container
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=face_color,
        edgecolor=border_color,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(box)

    # Top header bar inside box
    hdr_h = 0.44
    hdr = FancyBboxPatch(
        (x, y + h - hdr_h),
        w,
        hdr_h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=header_color,
        edgecolor=border_color,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(hdr)

    ax.text(
        x + w / 2.0,
        y + h - hdr_h / 2.0,
        title,
        ha="center",
        va="center",
        fontsize=8.9,
        fontweight="bold",
        color="#FFFFFF",
        zorder=4,
    )

    body_text = "\n".join(body_lines)
    ax.text(
        x + 0.14,
        y + h - hdr_h - 0.14,
        body_text,
        ha="left",
        va="top",
        fontsize=8.3,
        linespacing=1.35,
        color="#1F2937",
        zorder=4,
    )


def draw_gate_box(ax, x, y, w, h, title, body_lines):
    cx = x + w / 2.0
    # Chamfered octagonal decision-gate look for clean text fitting
    dx = 0.22
    dy = 0.22
    pts = [
        (x + dx, y),
        (x + w - dx, y),
        (x + w, y + dy),
        (x + w, y + h - dy),
        (x + w - dx, y + h),
        (x + dx, y + h),
        (x, y + h - dy),
        (x, y + dy),
    ]
    poly = Polygon(pts, closed=True, facecolor="#FFFBEB", edgecolor="#B45309", linewidth=1.8, zorder=2)
    ax.add_patch(poly)

    ax.text(
        cx,
        y + h - 0.28,
        title,
        ha="center",
        va="center",
        fontsize=9.1,
        fontweight="bold",
        color="#92400E",
        zorder=4,
    )
    ax.plot([x + 0.22, x + w - 0.22], [y + h - 0.50, y + h - 0.50], color="#F59E0B", lw=1.0, zorder=3)
    ax.text(
        cx,
        y + h - 0.64,
        "\n".join(body_lines),
        ha="center",
        va="top",
        fontsize=8.2,
        linespacing=1.35,
        color="#451A03",
        zorder=4,
    )


def arrow(ax, x1, y1, x2, y2, color="#334155", lw=1.5):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>,head_length=6.5,head_width=4.0",
        color=color,
        linewidth=lw,
        zorder=5,
    )
    ax.add_patch(arr)


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(15.6, 6.7), dpi=300)
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 6.7)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Top Title & Anti-Leakage Temporal Split Banner
    ax.text(
        7.8,
        6.40,
        "Diagram Alur Metodologi Eksperimen Bertahap (Zero Data Leakage Pipeline)",
        ha="center",
        va="center",
        fontsize=12.4,
        fontweight="bold",
        color="#0F172A",
    )

    banner = FancyBboxPatch(
        (0.25, 5.72),
        15.10,
        0.44,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#F1F5F9",
        edgecolor="#94A3B8",
        linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(banner)
    ax.text(
        7.8,
        5.94,
        "Split Kronologis:  Train (<= 2024-12-31, n=790) [Fit Scaler/PCA/SVD/Topic]  |  "
        "Val (2025, n=237) [Seleksi Fase 1-4]  |  Test (2026, n=154) [Evaluasi Blind]",
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color="#334155",
        zorder=2,
    )

    # Row 1 coordinates (4 equal boxes with generous width 3.45)
    r1_y = 3.22
    r1_h = 2.22
    box_w = 3.45
    gap_x = 0.43
    xs_r1 = [0.25 + i * (box_w + gap_x) for i in range(4)]

    draw_rounded_box(
        ax,
        xs_r1[0],
        r1_y,
        box_w,
        r1_h,
        "1. Data Acquisition & Alignment",
        [
            "- Kurs JISDOR USD/IDR (BI)",
            "  1.181 hari bursa -> target y_t",
            "- Berita Geopolitik CNBC (EN)",
            "  14.941 artikel (judul & isi)",
            "- Aturan Next Trading Day (WIB):",
            "  t_efektif > t_publikasi (100%)",
        ],
        header_color="#1E3A8A",
        face_color="#EFF6FF",
        border_color="#1E3A8A",
    )

    draw_rounded_box(
        ax,
        xs_r1[1],
        r1_y,
        box_w,
        r1_h,
        "2. Fitur Pasar (X_time) & Baseline",
        [
            "- 8 Indikator Teknikal Stasioner:",
            "  ret_1d, ret_2d, ret_3d, ret_5d",
            "  rate_to_ma5, rate_to_ma20",
            "  vol_5, vol_20 (<= hari t)",
            "- Membentuk Baseline M0_Market",
            "  (pembanding di Fase 1 s.d. 5)",
        ],
        header_color="#0F766E",
        face_color="#F0FDFA",
        border_color="#0F766E",
    )

    draw_rounded_box(
        ax,
        xs_r1[2],
        r1_y,
        box_w,
        r1_h,
        "3. Fase 1: Embeddings (17 Metode)",
        [
            "- Leksikal: BoW, N-Grams,",
            "  TF-IDF, BM25 (SVD Train-only)",
            "- Statis: Word2Vec, FastText, GloVe",
            "- Kontekstual: MiniLM, RoBERTa,",
            "  FLANG-RoBERTa, FinBERT",
            "- 6 Varian Hybrid -> Kunci Terbaik",
        ],
        header_color="#2563EB",
        face_color="#EFF6FF",
        border_color="#2563EB",
    )

    draw_rounded_box(
        ax,
        xs_r1[3],
        r1_y,
        box_w,
        r1_h,
        "4. Fase 2: Granulitas Teks",
        [
            "- Embedding Terkunci (Pemenang Fase 1):",
            "- Uji Granulitas Multimodal:",
            "  Judul Only vs Isi Only",
            "  vs Judul + Content",
            "- Uji Unimodal Teks (32d PCA)",
            "  untuk kontrol efek interaksi",
        ],
        header_color="#4338CA",
        face_color="#EEF2FF",
        border_color="#4338CA",
    )

    # Row 1 horizontal arrows
    mid_r1_y = r1_y + r1_h / 2.0
    for i in range(3):
        arrow(ax, xs_r1[i] + box_w + 0.03, mid_r1_y, xs_r1[i + 1] - 0.03, mid_r1_y)

    # Row 2 coordinates (left -> right)
    r2_y = 0.25
    r2_h = 2.25
    w5 = 3.70
    w6 = 2.95
    w7 = 3.70
    w8 = 3.45
    gap_r2 = 0.43
    x5 = 0.25
    x6 = x5 + w5 + gap_r2
    x7 = x6 + w6 + gap_r2
    x8 = x7 + w7 + gap_r2

    # Clean gutter connector from Box 4 (bottom) -> gutter -> Box 5 (top)
    gutter_y = 2.86
    src_x = xs_r1[3] + box_w / 2.0
    dst_x = x5 + w5 / 2.0
    ax.plot(
        [src_x, src_x, dst_x],
        [r1_y - 0.03, gutter_y, gutter_y],
        color="#334155",
        linewidth=1.5,
        zorder=4,
    )
    arrow(ax, dst_x, gutter_y, dst_x, r2_y + r2_h + 0.03)

    draw_rounded_box(
        ax,
        x5,
        r2_y,
        w5,
        r2_h,
        "5. Fase 3: Fitur Komplemen",
        [
            "- Uji Parsial, Tumpuk, & Unimodal:",
            "- Topic Modeling (Train-only):",
            "  NMF (16d) vs BERTopic(FinBERT) (16d)",
            "- Named Entity Recognition (NER):",
            "  spaCy (Struct 8d + SVD 16d) vs",
            "  Dictionary/Gazetteer Makro+FX (10d)",
        ],
        header_color="#7C3AED",
        face_color="#F5F3FF",
        border_color="#7C3AED",
    )

    draw_gate_box(
        ax,
        x6,
        r2_y,
        w6,
        r2_h,
        "6. Gate Seleksi (Val 2025)",
        [
            "Pilih kombinasi fitur terbaik",
            "berbasis Val Acc, F1, & AUC.",
            "Cegah overfitting dimensi",
            "sebelum masuk ke Fase 4.",
            "(Tanpa menyentuh Test 2026)",
        ],
    )

    draw_rounded_box(
        ax,
        x7,
        r2_y,
        w7,
        r2_h,
        "7. Fase 4: Strategi Fusion",
        [
            "- Gabung X_time & X_text Terkunci:",
            "- Early Fusion: Concat Awal",
            "  (MLP, LogisticRegression, LightGBM)",
            "- Late Fusion: Cabang Unimodal Terpisah",
            "  -> Kombinasi Probabilitas / Meta-LR",
            "- Intermediate: Dual Encoder (Attn+Gate)",
        ],
        header_color="#B91C1C",
        face_color="#FEF2F2",
        border_color="#B91C1C",
    )

    draw_rounded_box(
        ax,
        x8,
        r2_y,
        w8,
        r2_h,
        "8. Fase 5: Evaluasi & Analisis",
        [
            "- Uji Ablasi (Val 2025 & Test 2026):",
            "  Model A (T2) vs B (T2 NLP) vs C (T3)",
            "- Metrik Konsisten: Acc, Macro-F1, AUC",
            "- UMAP Volatilitas Tinggi vs Rendah",
            "- Permutation Imp. & Error Analysis",
            "  (Tanggal NLP Saves vs Noise)",
        ],
        header_color="#1F2937",
        face_color="#F8FAFC",
        border_color="#1F2937",
    )

    mid_r2_y = r2_y + r2_h / 2.0
    arrow(ax, x5 + w5 + 0.03, mid_r2_y, x6 - 0.03, mid_r2_y)
    arrow(ax, x6 + w6 + 0.03, mid_r2_y, x7 - 0.03, mid_r2_y)
    arrow(ax, x7 + w7 + 0.03, mid_r2_y, x8 - 0.03, mid_r2_y)

    out_path = Path("NLP-Project/reports/figures/diagram_alur_eksperimen_bab2.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    art_dir = Path(r"C:\Users\LENOVO\.gemini\antigravity\brain\001877a4-72ea-408f-a9be-9eddf6b2acc0")
    shutil.copy2(out_path, art_dir / out_path.name)
    print(f"Saved diagram -> {out_path} & artifact copy")


if __name__ == "__main__":
    main()
