import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from src.metrics import format_mean_std

# ======================================================================
# 10. Result table
# ======================================================================
def build_result_table(fold_results):
    rows = []

    for r in fold_results:
        rows.append({
            "Fold": r["fold"],
            "AUC": r["AUC"],
            "ACC": r["ACC"],
            "F1": r["F1"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "Best Epoch": r["best_epoch"],
            "Top Epochs": ",".join(map(str, r["top_epochs"])) if len(r["top_epochs"]) > 0 else "-",
            "Best Threshold": r["best_threshold"],
            "Best Inner Val AUC": r["best_inner_val_auc"],
            "Best Inner Val F1@Thr": r["best_inner_val_f1_at_thr"],
            "Top3 Inner Val AUC Mean": r["topk_inner_val_auc_mean"],
            "Train N": r["n_train"],
            "Val N": r["n_val"],
            "Test N": r["n_test"]
        })

    df = pd.DataFrame(rows)

    summary_row = {
        "Fold": "Mean ± Std",
        "AUC": format_mean_std(df["AUC"]),
        "ACC": format_mean_std(df["ACC"]),
        "F1": format_mean_std(df["F1"]),
        "Precision": format_mean_std(df["Precision"]),
        "Recall": format_mean_std(df["Recall"]),
        "Best Epoch": "-",
        "Top Epochs": "-",
        "Best Threshold": format_mean_std(df["Best Threshold"]),
        "Best Inner Val AUC": format_mean_std(df["Best Inner Val AUC"]),
        "Best Inner Val F1@Thr": format_mean_std(df["Best Inner Val F1@Thr"]),
        "Top3 Inner Val AUC Mean": format_mean_std(df["Top3 Inner Val AUC Mean"]),
        "Train N": "-",
        "Val N": "-",
        "Test N": "-"
    }

    df_show = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    print("\n" + "#" * 80)
    print("Outer 5-Fold Test Results")
    print("#" * 80)
    print(df_show.to_string(index=False))

    return df_show


# ======================================================================
# 11. Save result table and plot ROC, PRC, and Confusion Matrix
# ======================================================================
def save_results_and_plots(
    fold_results,
    save_dir,
    model_tag,
    image_feat_weight,
    struct_feat_weight,
):
    os.makedirs(save_dir, exist_ok=True)

    df_show = build_result_table(fold_results)

    result_csv = os.path.join(save_dir, f"{model_tag}_results.csv")
    df_show.to_csv(result_csv, index=False, encoding="utf-8-sig")
    print("结果表格已保存:", result_csv)

    y_true_all = np.concatenate([r["y_true"] for r in fold_results])
    y_prob_all = np.concatenate([r["y_prob"] for r in fold_results])
    y_pred_all = np.concatenate([r["y_pred"] for r in fold_results])

    # ======================================================================
    # 12. Plot a single ROC curve using concatenated 5-fold outer-test predictions and save ROC csv
    # ======================================================================
    fpr, tpr, thresholds = roc_curve(y_true_all, y_prob_all)
    roc_auc = roc_auc_score(y_true_all, y_prob_all)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=(
            f"{model_tag} "
            f"(image={image_feat_weight:.2f}, struct={struct_feat_weight:.2f}, "
            f"AUC = {roc_auc:.4f})"
        )
    )
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    })

    roc_csv = os.path.join(save_dir, f"{model_tag}_roc.csv")
    roc_df.to_csv(roc_csv, index=False, encoding="utf-8-sig")
    print("ROC csv 已保存:", roc_csv)

    # ======================================================================
    # 13. Plot a single PRC curve using concatenated 5-fold outer-test predictions and save PRC csv
    # ======================================================================
    precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)
    ap = average_precision_score(y_true_all, y_prob_all)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        lw=2,
        label=(
            f"{model_tag} "
            f"(image={image_feat_weight:.2f}, struct={struct_feat_weight:.2f}, "
            f"AP = {ap:.4f})"
        )
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    prc_df = pd.DataFrame({
        "recall": recall,
        "precision": precision
    })

    prc_csv = os.path.join(save_dir, f"{model_tag}_prc.csv")
    prc_df.to_csv(prc_csv, index=False, encoding="utf-8-sig")
    print("PRC csv 已保存:", prc_csv)

    # ======================================================================
    # 14. Plot Confusion Matrix for the five outer folds
    # ======================================================================
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, r in enumerate(fold_results):
        ax = axes[i]
        cm = r["cm"]

        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Outer Fold {r['fold']}\nAUC={r['AUC']:.3f}, ACC={r['ACC']:.3f}")
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")

        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(
                    col,
                    row,
                    str(cm[row, col]),
                    ha="center",
                    va="center",
                    color="black"
                )

    for j in range(len(fold_results), len(axes)):
        axes[j].axis("off")

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    plt.suptitle(
        f"Confusion Matrices of Outer 5-Fold Test Results\n"
        f"{model_tag} "
        f"(image={image_feat_weight:.2f}, struct={struct_feat_weight:.2f})",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

    return df_show
