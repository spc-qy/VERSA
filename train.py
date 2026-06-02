# ======================================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

from configs.config import (
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    DEVICE,
    N_SPLITS,
    CV_RANDOM_STATE,
    INNER_VAL_RATIO,
    STAGE1_EPOCHS,
    LR_STAGE1,
    LR_STAGE2,
    TOP_K,
    IMAGE_FEAT_WEIGHT,
    STRUCT_FEAT_WEIGHT,
    USE_STRUCT_BRANCH,
    USE_PRIOR_FOR_VAL_TEST,
    SAVE_DIR,
)
from src.heatmap_features import STRUCT_FEAT_DIM
from src.dataset import (
    collect,
    TwoViewSpineDataset_ImageOnly,
    TwoViewSpineDataset_WithStruct,
)
from src.model import (
    TwoViewNetB_ImageOnly,
    TwoViewNetB_Fusion,
    freeze_backbone_except_layer4_imageonly,
    freeze_backbone_except_layer4_fusion,
    unfreeze_all_backbone,
)
from src.train_utils import (
    set_seed,
    run_epoch_imageonly,
    run_epoch_fusion,
    average_state_dicts_original,
    average_state_dicts_safe,
)
from src.metrics import calc_metrics, find_best_threshold
from src.plot_utils import save_results_and_plots


def main():
    print("Using device:", DEVICE)
    print("Outer 5-fold CV:", N_SPLITS)
    print("Nested split: outer-train -> inner-train + inner-val")
    print("Data augmentation: only applied on inner-train")
    print(f"Two-stage finetuning: stage1={STAGE1_EPOCHS} epochs, lr1={LR_STAGE1}, lr2={LR_STAGE2}")
    print("Thresholding: choose best threshold on inner-val instead of fixed 0.5")
    print("BN handling: keep backbone BatchNorm in eval mode during training")
    print(f"Model selection: keep top-{TOP_K} epochs by inner val AUC, then average checkpoints")
    print("Hard switch mode:")
    print("  IMAGE_FEAT_WEIGHT = 1.0 -> use code2 original image-only path")
    print("  IMAGE_FEAT_WEIGHT < 1.0 -> use 512 image + 512 structure fusion path")
    print(f"Fusion weight: image={IMAGE_FEAT_WEIGHT:.2f}, structure={STRUCT_FEAT_WEIGHT:.2f}")
    print("USE_STRUCT_BRANCH:", USE_STRUCT_BRANCH)
    print("USE_PRIOR_FOR_VAL_TEST:", USE_PRIOR_FOR_VAL_TEST)

    set_seed(CV_RANDOM_STATE)

    print("Heatmap structural feature dim:", STRUCT_FEAT_DIM)

    # ======================================================================
    # 4. Sample collection
    # ======================================================================
    samples = collect("AB", 0) + collect("BA", 1)
    labels_all = [s[4] for s in samples]

    print("Total samples:", len(samples))
    print("Class 0 count:", sum(1 for s in samples if s[4] == 0))
    print("Class 1 count:", sum(1 for s in samples if s[4] == 1))

    if len(samples) == 0:
        raise RuntimeError("没有收集到样本，请检查 DATA_ROOT / PRIORS_ROOT / 类别名 / 文件名。")

    # ======================================================================
    # 9. Outer 5-fold + inner validation
    # ======================================================================
    outer_skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE
    )

    fold_results = []

    for fold, (outer_trainval_idx, outer_test_idx) in enumerate(
        outer_skf.split(samples, labels_all),
        start=1
    ):
        print("\n" + "=" * 80)
        print(f"Outer Fold {fold}/{N_SPLITS}")
        print("=" * 80)

        outer_trainval_samples = [samples[i] for i in outer_trainval_idx]
        outer_trainval_labels = [labels_all[i] for i in outer_trainval_idx]
        outer_test_samples = [samples[i] for i in outer_test_idx]

        inner_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=INNER_VAL_RATIO,
            random_state=CV_RANDOM_STATE + fold
        )

        inner_train_rel_idx, inner_val_rel_idx = next(
            inner_splitter.split(outer_trainval_samples, outer_trainval_labels)
        )

        inner_train_samples = [outer_trainval_samples[i] for i in inner_train_rel_idx]
        inner_val_samples   = [outer_trainval_samples[i] for i in inner_val_rel_idx]

        print(f"Inner train size: {len(inner_train_samples)}")
        print(f"Inner val size:   {len(inner_val_samples)}")
        print(f"Outer test size:  {len(outer_test_samples)}")

        # --------------------------------------------------------------
        # Hard switch of Dataset / Model / run_epoch according to IMAGE_FEAT_WEIGHT
        # --------------------------------------------------------------
        if not USE_STRUCT_BRANCH:
            train_ds = TwoViewSpineDataset_ImageOnly(inner_train_samples, use_prior=True, train=True)
            val_ds   = TwoViewSpineDataset_ImageOnly(inner_val_samples, use_prior=USE_PRIOR_FOR_VAL_TEST, train=False)
            test_ds  = TwoViewSpineDataset_ImageOnly(outer_test_samples, use_prior=USE_PRIOR_FOR_VAL_TEST, train=False)

            model = TwoViewNetB_ImageOnly().to(DEVICE)
            freeze_backbone_except_layer4_imageonly(model)
            run_epoch_fn = run_epoch_imageonly

        else:
            train_ds = TwoViewSpineDataset_WithStruct(inner_train_samples, use_prior=True, train=True)
            val_ds   = TwoViewSpineDataset_WithStruct(inner_val_samples, use_prior=USE_PRIOR_FOR_VAL_TEST, train=False)
            test_ds  = TwoViewSpineDataset_WithStruct(outer_test_samples, use_prior=USE_PRIOR_FOR_VAL_TEST, train=False)

            model = TwoViewNetB_Fusion(image_weight=IMAGE_FEAT_WEIGHT).to(DEVICE)
            freeze_backbone_except_layer4_fusion(model)
            run_epoch_fn = run_epoch_fusion

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_STAGE1
        )

        best_val_auc = -1.0
        best_epoch = -1
        best_threshold = 0.5
        best_val_f1_at_thr = -1.0

        top_models = []  # [(val_auc, epoch, threshold, val_f1_at_thr, state_dict), ...]

        for ep in range(1, EPOCHS + 1):

            # Uncomment the following lines to enable the second fine-tuning stage
            # if ep == STAGE1_EPOCHS + 1:
            #     unfreeze_all_backbone(model)
            #     optimizer = torch.optim.AdamW(model.parameters(), lr=LR_STAGE2)

            tr_loss, tr_prob, tr_pred, tr_y = run_epoch_fn(
                model,
                train_dl,
                criterion,
                optimizer,
                train=True
            )

            va_loss, va_prob, va_pred, va_y = run_epoch_fn(
                model,
                val_dl,
                criterion,
                optimizer=None,
                train=False
            )

            try:
                tr_auc = roc_auc_score(tr_y, tr_prob)
            except Exception:
                tr_auc = np.nan

            try:
                va_auc = roc_auc_score(va_y, va_prob)
            except Exception:
                va_auc = np.nan

            cur_thr, cur_val_f1 = find_best_threshold(va_y, va_prob)

            print(
                f"[outer fold {fold:02d} | ep {ep:03d}] "
                f"train_loss={tr_loss:.4f}, train_auc={tr_auc:.4f} | "
                f"inner_val_loss={va_loss:.4f}, inner_val_auc={va_auc:.4f}, "
                f"best_thr={cur_thr:.3f}, inner_val_f1@thr={cur_val_f1:.4f}"
            )

            if not np.isnan(va_auc):
                state_now = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                top_models.append((va_auc, ep, cur_thr, cur_val_f1, state_now))
                top_models = sorted(top_models, key=lambda x: x[0], reverse=True)[:TOP_K]

            if (not np.isnan(va_auc)) and (va_auc > best_val_auc):
                best_val_auc = va_auc
                best_epoch = ep
                best_threshold = cur_thr
                best_val_f1_at_thr = cur_val_f1

        print(
            f"[outer fold {fold:02d}] Best epoch (from inner val) = {best_epoch}, "
            f"best inner val AUC = {best_val_auc:.4f}, "
            f"selected threshold = {best_threshold:.3f}, "
            f"inner val F1@thr = {best_val_f1_at_thr:.4f}"
        )

        top_epochs = [x[1] for x in top_models]
        top_aucs = [x[0] for x in top_models]
        top_thresholds = [x[2] for x in top_models]

        print(f"[outer fold {fold:02d}] Top-{TOP_K} epochs = {top_epochs}")
        print(f"[outer fold {fold:02d}] Top-{TOP_K} inner val AUCs = {[round(x, 4) for x in top_aucs]}")
        print(f"[outer fold {fold:02d}] Top-{TOP_K} thresholds = {[round(x, 3) for x in top_thresholds]}")

        if len(top_models) > 0:
            if not USE_STRUCT_BRANCH:
                avg_state = average_state_dicts_original([x[4] for x in top_models])
            else:
                avg_state = average_state_dicts_safe([x[4] for x in top_models])

            model.load_state_dict(avg_state)

            # Use the average threshold from the top-k models to avoid relying on a single epoch
            best_threshold = float(np.mean([x[2] for x in top_models]))
            best_val_f1_at_thr = float(np.mean([x[3] for x in top_models]))

        test_loss, test_prob, test_pred_default, test_y = run_epoch_fn(
            model,
            test_dl,
            criterion,
            optimizer=None,
            train=False
        )

        test_pred = (test_prob >= best_threshold).astype(np.int64)

        metrics = calc_metrics(test_y, test_prob, test_pred)

        print(
            f"[outer fold {fold:02d}] "
            f"TEST AUC={metrics['AUC']:.4f}, "
            f"ACC={metrics['ACC']:.4f}, "
            f"F1={metrics['F1']:.4f}, "
            f"Precision={metrics['Precision']:.4f}, "
            f"Recall={metrics['Recall']:.4f}, "
            f"Threshold={best_threshold:.3f}"
        )

        fold_results.append({
            "fold": fold,
            "n_train": len(inner_train_samples),
            "n_val": len(inner_val_samples),
            "n_test": len(outer_test_samples),
            "best_epoch": best_epoch,
            "top_epochs": top_epochs,
            "best_threshold": best_threshold,
            "best_inner_val_auc": best_val_auc,
            "best_inner_val_f1_at_thr": best_val_f1_at_thr,
            "topk_inner_val_auc_mean": float(np.mean(top_aucs)) if len(top_aucs) > 0 else np.nan,
            "AUC": metrics["AUC"],
            "ACC": metrics["ACC"],
            "F1": metrics["F1"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "y_true": test_y,
            "y_prob": test_prob,
            "y_pred": test_pred,
            "cm": metrics["CM"]
        })

    if not USE_STRUCT_BRANCH:
        model_tag = "image_weight_1_code2_imageonly_path"
    else:
        model_tag = f"weighted_img{IMAGE_FEAT_WEIGHT:.2f}_struct{STRUCT_FEAT_WEIGHT:.2f}_512_512_fusion"

    save_results_and_plots(
        fold_results=fold_results,
        save_dir=SAVE_DIR,
        model_tag=model_tag,
        image_feat_weight=IMAGE_FEAT_WEIGHT,
        struct_feat_weight=STRUCT_FEAT_WEIGHT,
    )


if __name__ == "__main__":
    main()
