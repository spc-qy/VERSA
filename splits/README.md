# Outer-fold assignments

This directory contains the exact animal-level outer-fold assignments used in
the reported benchmark analyses.

## two_view_AB_vs_BA_100perclass_outer_folds.csv

This file contains the exact five-fold outer-test assignments for the balanced
two-view benchmark comparing T13L7 (AB) and T14L6 (BA), with 100 animals per
class.

The split was generated using stratified five-fold cross-validation with
`shuffle=True` and `random_state=42`.

Each animal appears in exactly one outer-test fold.
