# Model Checkpoints

This directory provides information on the model checkpoints associated with
the reported VERSA benchmark analysis.

## Two-view balanced benchmark

The checkpoint archive contains the final models corresponding to the five
held-out outer folds of the balanced two-view benchmark comparing T13L7 (AB)
and T14L6 (BA), with 100 animals per class.

Each checkpoint contains the final top-k averaged model state used for
outer-test evaluation, together with the corresponding cross-validation
metadata and decision threshold.

## Download

The checkpoint archive is available from:

http://bioinfor.imu.edu.cn/biocloud/downloads/VERSA_two_view_AB_vs_BA_100perclass_checkpoints_v1.0.0.tar.gz

## Version

VERSA code version: **1.0.0**

## Corresponding outer-fold assignments

The exact outer-fold assignments are available in:

[`../splits/two_view_AB_vs_BA_100perclass_outer_folds.csv`](../splits/two_view_AB_vs_BA_100perclass_outer_folds.csv)

## Checkpoint files

The archive contains:

- `two_view_AB_vs_BA_100perclass_image_weight_1.00_imageonly_path_outer_fold1.pt`
- `two_view_AB_vs_BA_100perclass_image_weight_1.00_imageonly_path_outer_fold2.pt`
- `two_view_AB_vs_BA_100perclass_image_weight_1.00_imageonly_path_outer_fold3.pt`
- `two_view_AB_vs_BA_100perclass_image_weight_1.00_imageonly_path_outer_fold4.pt`
- `two_view_AB_vs_BA_100perclass_image_weight_1.00_imageonly_path_outer_fold5.pt`

## SHA256

`e4bb33ebdb58bde7dcc42266dba1c11bfc2882b5b64d0cb8b255c98408f3921e`
