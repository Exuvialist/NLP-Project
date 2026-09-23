"""
Module: evaluate.py
Deskripsi: Modul evaluasi untuk menghitung F1-score (Macro, Class 0, Class 1)
serta Confusion Matrix sesuai ketentuan metrik proyek.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    split_name: str = "Validation",
) -> Dict[str, Any]:
    """
    Menghitung F1-score dan Confusion Matrix.

    Parameters
    ----------
    y_true : np.ndarray
        Nilai target sebenarnya (0 atau 1).
    y_pred : np.ndarray
        Nilai prediksi kelas biner (0 atau 1).
    model_name : str
        Nama model yang dievaluasi.
    split_name : str
        Nama split data (Train / Validation / Test).

    Returns
    -------
    Dict[str, Any]
        Dictionary berisi metrik F1-score dan confusion matrix.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_down = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_up = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "model": model_name,
        "split": split_name,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_down": f1_down,
        "f1_up": f1_up,
        "accuracy": acc,
        "confusion_matrix": cm,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    }


def print_evaluation(eval_dict: Dict[str, Any]) -> None:
    """Mencetak ringkasan evaluasi F1-Score dan Confusion Matrix ke terminal."""
    cm = eval_dict["confusion_matrix"]
    print(f"\n{'='*50}")
    print(f"EVALUASI: {eval_dict['model']} | Split: {eval_dict['split']}")
    print(f"{'='*50}")
    print(f"• F1-Score (Macro)   : {eval_dict['f1_macro']:.4f}")
    print(f"• F1-Score (Weighted): {eval_dict['f1_weighted']:.4f}")
    print(f"• F1-Score (Class DOWN/0): {eval_dict['f1_down']:.4f}")
    print(f"• F1-Score (Class UP/1)  : {eval_dict['f1_up']:.4f}")
    print(f"• Accuracy           : {eval_dict['accuracy']:.4f}")
    print("\nConfusion Matrix (Label: [0=DOWN, 1=NAIK]):")
    print(f"               Pred DOWN (0)   Pred UP (1)")
    print(f"Actual DOWN (0)    {cm[0, 0]:5d}          {cm[0, 1]:5d}")
    print(f"Actual UP (1)      {cm[1, 0]:5d}          {cm[1, 1]:5d}")
    print(f"{'='*50}")


def build_comparison_dataframe(eval_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Mengubah daftar dictionary hasil evaluasi menjadi DataFrame perbandingan."""
    records = []
    for res in eval_results:
        records.append({
            "Model": res["model"],
            "Split": res["split"],
            "F1 Macro": round(res["f1_macro"], 4),
            "F1 Weighted": round(res["f1_weighted"], 4),
            "F1 DOWN (0)": round(res["f1_down"], 4),
            "F1 UP (1)": round(res["f1_up"], 4),
            "Accuracy": round(res["accuracy"], 4),
            "Confusion Matrix [TN, FP, FN, TP]": f"[{res['tn']}, {res['fp']}, {res['fn']}, {res['tp']}]",
        })
    return pd.DataFrame(records)

