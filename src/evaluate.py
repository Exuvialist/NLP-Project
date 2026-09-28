from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score, roc_curve

from src.config import (
    DAILY_ALIGNED_CSV,
    FIGURES_DIR,
    PREDICTIONS_TEST_CSV,
    PREDICTIONS_VALIDATION_CSV,
    RESULTS_TEST_CSV,
    RESULTS_VALIDATION_CSV,
    TARGET_COLUMN,
    TEST_CSV,
    ensure_output_dirs,
)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "directional_accuracy": accuracy_score(y_true, y_pred),
        "f1_up": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"model": model, "n": len(group), **compute_metrics(group["y_true"], group["y_pred"], group["y_proba"])}
        for model, group in predictions.groupby("model", sort=False)
    ]
    return pd.DataFrame(rows)


def plot_metrics_comparison(results_test: pd.DataFrame) -> None:
    model_results = results_test[results_test["model"].str.startswith("model")].reset_index(drop=True)
    x = np.arange(len(model_results))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, metric, label, color in [
        (-width, "directional_accuracy", "Akurasi arah", "#4c72b0"),
        (0.0, "f1_up", "F1 naik", "#55a868"),
        (width, "f1_macro", "F1 macro", "#dd8452"),
    ]:
        values = model_results[metric].to_numpy()
        ax.bar(x + offset, values, width, label=label, color=color)
        for xi, value in zip(x + offset, values):
            if np.isfinite(value):
                ax.text(xi, value + 0.008, f"{value:.3f}", ha="center", fontsize=7)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Acak (0,5)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_results["model"], rotation=12)
    ax.set_ylim(0.0, 0.85)
    ax.set_ylabel("Nilai metrik")
    ax.set_title("Perbandingan akurasi arah dan F1 pada test 2026 (arah USD/IDR)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_comparison_test.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(predictions_test: pd.DataFrame) -> None:
    model_names = [name for name in predictions_test["model"].unique() if name.startswith("model")]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, model_name in zip(axes.ravel(), model_names):
        group = predictions_test[predictions_test["model"] == model_name]
        matrix = confusion_matrix(group["y_true"], group["y_pred"], labels=[0, 1])
        ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], labels=["Pred turun", "Pred naik"])
        ax.set_yticks([0, 1], labels=["Aktual turun", "Aktual naik"])
        ax.set_title(model_name, fontsize=9)
    fig.suptitle("Confusion matrix pada test 2026")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_test.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(predictions_test: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for model_name, group in predictions_test.groupby("model", sort=False):
        if not model_name.startswith("model"):
            continue
        fpr, tpr, _ = roc_curve(group["y_true"], group["y_proba"])
        auc_value = roc_auc_score(group["y_true"], group["y_proba"])
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Kurva ROC pada test 2026")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_test.png", dpi=150)
    plt.close(fig)


def plot_predictions_vs_actual(predictions_test: pd.DataFrame) -> None:
    test = pd.read_csv(TEST_CSV, parse_dates=["trading_date"])
    predictions_test = predictions_test.copy()
    predictions_test["trading_date"] = pd.to_datetime(predictions_test["trading_date"])

    fig = plt.figure(figsize=(12, 10))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.35, 1, 1])
    ax_top = fig.add_subplot(grid[0, :])
    ax_top.plot(test["trading_date"], test["usd_idr_rate"], color="#333333", linewidth=1.5, label="Kurs USD/IDR (aktual)")
    model3 = predictions_test[predictions_test["model"] == "model3_market_sentiment_tfidf"].sort_values("trading_date")
    bottom = float(test["usd_idr_rate"].min()) * 0.998
    top = float(test["usd_idr_rate"].max()) * 1.002
    ax_top.fill_between(
        model3["trading_date"], bottom, top, where=model3["y_pred"].to_numpy() == 1,
        color="#55a868", alpha=0.14, label="Prediksi naik (model3)",
    )
    ax_top.fill_between(
        model3["trading_date"], bottom, top, where=model3["y_pred"].to_numpy() == 0,
        color="#c44e52", alpha=0.12, label="Prediksi turun (model3)",
    )
    ax_top.set_ylim(bottom, top)
    ax_top.set_ylabel("IDR per USD")
    ax_top.set_title("Overlay prediksi arah dengan kurs aktual pada test 2026")
    ax_top.legend(fontsize=8, loc="upper left")

    daily = pd.read_csv(DAILY_ALIGNED_CSV, parse_dates=["trading_date"])
    daily["next_return"] = daily["usd_idr_rate"].shift(-1) / daily["usd_idr_rate"] - 1.0

    model_names = [name for name in predictions_test["model"].unique() if name.startswith("model")]
    direction_axes = [fig.add_subplot(grid[row, col]) for row, col in [(1, 0), (1, 1), (2, 0), (2, 1)]]
    legend_handles = [
        Patch(facecolor="#d9d9d9", label="Return aktual (%)"),
        Line2D([], [], marker="^", color="none", markerfacecolor="#55a868", markersize=7, label="Prediksi naik benar"),
        Line2D([], [], marker="^", color="none", markerfacecolor="#c44e52", markersize=7, label="Prediksi naik salah"),
        Line2D([], [], marker="v", color="none", markerfacecolor="#55a868", markersize=7, label="Prediksi turun benar"),
        Line2D([], [], marker="v", color="none", markerfacecolor="#c44e52", markersize=7, label="Prediksi turun salah"),
    ]
    for ax, model_name in zip(direction_axes, model_names):
        group = predictions_test[predictions_test["model"] == model_name].sort_values("trading_date")
        group = group.merge(daily[["trading_date", "next_return"]], on="trading_date", how="left")
        returns = group["next_return"].to_numpy() * 100.0
        correct = (group["y_true"] == group["y_pred"]).to_numpy()
        ax.bar(group["trading_date"], returns, width=1.2, color="#d9d9d9", label="Return aktual (%)")
        ax.axhline(0, color="#777777", linewidth=0.8)
        for marker, mask in [("^", group["y_pred"].to_numpy() == 1), ("v", group["y_pred"].to_numpy() == 0)]:
            ax.scatter(
                group.loc[mask, "trading_date"], returns[mask], marker=marker, s=26,
                c=np.where(correct[mask], "#55a868", "#c44e52"), edgecolors="white", linewidths=0.4, zorder=3,
            )
        accuracy = accuracy_score(group["y_true"], group["y_pred"])
        f1_macro = f1_score(group["y_true"], group["y_pred"], average="macro", zero_division=0)
        limit = max(float(np.nanmax(np.abs(returns))) * 1.35, 0.5)
        ax.set_ylim(-limit, limit)
        ax.set_ylabel("Return (%)", fontsize=8)
        ax.set_title(f"{model_name} (akurasi={accuracy:.3f}, F1 macro={f1_macro:.3f})", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(handles=legend_handles, fontsize=6.3, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prediksi_vs_aktual_test.png", dpi=150)
    plt.close(fig)


def draw_box(ax, x, y, width, height, text, facecolor="#eef3fa", edgecolor="#4c72b0", fontsize=9, dashed=False) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.08",
        linewidth=1.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def draw_note(ax, x, y, text, color="#8a6d3b", fontsize=7.5) -> None:
    ax.text(x, y, text, ha="center", va="top", fontsize=fontsize, style="italic", color=color)


def draw_arrow(ax, start, end) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#555555", linewidth=1.1)
    ax.add_patch(arrow)


def plot_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(14, 6.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(2.3, 8.6)
    ax.axis("off")

    # --- kolom 1: sumber data ---
    ax.text(1.5, 8.15, "Sumber data", fontsize=10, style="italic", color="#555555", ha="center")
    draw_box(ax, 0.15, 6.85, 2.7, 0.95, "Kurs JISDOR BI\n1.202 hari perdagangan\n2021-09-01 s.d. 2026-09-01")
    draw_box(ax, 0.15, 5.55, 2.7, 0.95, "Berita geopolitik CNBC\n14.941 artikel (hasil cleaning Tugas 1)", facecolor="#fdf3e7", edgecolor="#dd8452")

    # --- kolom 2: fitur ---
    ax.text(4.8, 8.15, "Fitur", fontsize=10, style="italic", color="#555555", ha="center")
    draw_box(ax, 3.15, 6.85, 3.3, 1.1, "Fitur pasar (9)\nret 1/2/3/5d - rate/MA5, MA20\nvol 5/20 - log_rate", facecolor="#fdf3e7", edgecolor="#dd8452")
    draw_note(ax, 4.8, 6.72, "jendela berakhir di t (kurs <= t)")
    draw_box(ax, 3.15, 5.30, 3.3, 1.15, "Sentimen (17)\nVADER: mean/median/std, rasio pos/neg/netral\nLM: polaritas, neg/pos/unc/modal per 1k token", facecolor="#e8f4ec", edgecolor="#55a868")
    draw_note(ax, 4.8, 5.17, "agregasi harian per artikel")
    draw_box(ax, 3.15, 3.85, 3.3, 1.15, "TF-IDF + SVD (50)\n5.000 fitur, min_df=3, ngram 1-2, sublinear_tf\nTruncatedSVD -> StandardScaler", facecolor="#e8f4ec", edgecolor="#55a868")
    draw_note(ax, 4.8, 3.72, "vocabulary/IDF/SVD fit hanya di TRAIN", color="#a94442")

    # --- integrasi + split ---
    draw_box(ax, 6.95, 5.30, 2.9, 1.15, "Integrasi harian\n1 baris = 1 hari perdagangan\ncutoff: hanya berita terbit < t")
    draw_box(ax, 6.95, 3.85, 2.9, 0.95, "Split kronologis\ntrain <= 2024-12 | val 2025\ntest 2026-01 s.d. 2026-08", facecolor="#f2e8f7", edgecolor="#8172b3")

    # --- kolom 3: model & evaluasi ---
    ax.text(12.05, 8.15, "Model & evaluasi", fontsize=10, style="italic", color="#555555", ha="center")
    draw_box(ax, 10.30, 6.85, 1.70, 0.85, "model0\npasar (9)")
    draw_box(ax, 12.20, 6.85, 1.70, 0.85, "model1\n+ sentimen (26)")
    draw_box(ax, 10.30, 5.80, 1.70, 0.85, "model2\n+ TF-IDF (59)")
    draw_box(ax, 12.20, 5.80, 1.70, 0.85, "model3\n+ keduanya (76)")
    draw_note(ax, 12.05, 5.65, "LogisticRegression + StandardScaler\npemilihan text source TF-IDF hanya di validation", color="#555555")
    draw_box(ax, 10.30, 3.85, 3.60, 1.00, "Evaluasi\nakurasi arah - F1 up/macro - MCC\nROC AUC - confusion matrix", facecolor="#f2e8f7", edgecolor="#8172b3")
    draw_box(ax, 10.30, 2.75, 3.60, 0.70, "Baseline naif: all-up - persistence", facecolor="#f7f7f7", edgecolor="#999999", fontsize=8)

    # --- guardrail anti-leakage ---
    draw_box(ax, 0.15, 2.75, 5.95, 1.00,
             "Guardrail anti-leakage\nfitur kurs hanya <= t - berita hanya yang terbit sebelum hari efektifnya\nTF-IDF/SVD/scaler di-fit pada train saja - test dievaluasi sekali",
             facecolor="#fff8f0", edgecolor="#a94442", dashed=True, fontsize=8)

    # --- panah ---
    draw_arrow(ax, (2.85, 7.37), (3.15, 7.42))
    draw_arrow(ax, (2.85, 6.10), (3.15, 5.92))
    draw_arrow(ax, (2.85, 5.85), (3.15, 4.55))
    draw_arrow(ax, (6.45, 7.30), (6.95, 6.20))
    draw_arrow(ax, (6.45, 5.90), (6.95, 5.90))
    draw_arrow(ax, (6.45, 4.50), (6.95, 5.58))
    draw_arrow(ax, (8.40, 5.30), (8.40, 4.80))
    draw_arrow(ax, (9.85, 4.40), (10.30, 6.10))
    draw_arrow(ax, (12.10, 5.80), (12.10, 4.85))

    ax.set_title("Arsitektur pipeline Tugas 2: prediksi arah USD/IDR dengan fitur pasar + NLP", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pipeline_tugas2.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    predictions_validation = pd.read_csv(PREDICTIONS_VALIDATION_CSV)
    predictions_test = pd.read_csv(PREDICTIONS_TEST_CSV)

    results_validation = summarize_predictions(predictions_validation)
    results_test = summarize_predictions(predictions_test)
    results_validation.to_csv(RESULTS_VALIDATION_CSV, index=False)
    results_test.to_csv(RESULTS_TEST_CSV, index=False)

    # Print ringkasan hanya model0-model3; results_*.csv tetap memuat baseline naif.
    print("hasil validasi 2025 (model0-model3):")
    print(results_validation[results_validation["model"].str.startswith("model")].round(4).to_string(index=False))
    print("hasil test 2026 (model0-model3):")
    print(results_test[results_test["model"].str.startswith("model")].round(4).to_string(index=False))

    plot_metrics_comparison(results_test)
    plot_confusion_matrices(predictions_test)
    plot_roc_curves(predictions_test)
    plot_predictions_vs_actual(predictions_test)
    plot_pipeline_diagram()
    print(f"figure evaluasi -> {FIGURES_DIR}")
    print(f"tabel hasil -> {RESULTS_VALIDATION_CSV.name}, {RESULTS_TEST_CSV.name}")


if __name__ == "__main__":
    main()
