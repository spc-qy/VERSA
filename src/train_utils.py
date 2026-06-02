import copy
import random

import numpy as np
import torch

from configs.config import DEVICE
from src.model import set_backbone_bn_eval

# ======================================================================
# 2.1 Reproducibility: set random seed
# ======================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # For stronger determinism, you may try enabling the following lines
    # torch.use_deterministic_algorithms(True)
    # torch.set_deterministic_debug_mode("warn")

# ======================================================================
# 6.3 Helper functions for top-k checkpoint averaging
# ======================================================================
def average_state_dicts_original(state_dicts):
    """
    Original top-k checkpoint averaging from code 2.
    This version must be used when IMAGE_FEAT_WEIGHT = 1.0 to maximally reproduce code 2.
    """
    if len(state_dicts) == 0:
        raise ValueError("state_dicts is empty")

    avg_state = copy.deepcopy(state_dicts[0])

    for k in avg_state.keys():
        for i in range(1, len(state_dicts)):
            avg_state[k] = avg_state[k] + state_dicts[i][k]
        avg_state[k] = avg_state[k] / len(state_dicts)

    return avg_state


def average_state_dicts_safe(state_dicts):
    if len(state_dicts) == 0:
        raise ValueError("state_dicts is empty")

    avg_state = copy.deepcopy(state_dicts[0])

    for k in avg_state.keys():
        if torch.is_floating_point(avg_state[k]):
            avg_state[k] = sum(sd[k].float() for sd in state_dicts) / len(state_dicts)
        else:
            avg_state[k] = state_dicts[0][k]

    return avg_state

# ======================================================================
# 7A.
# ======================================================================
def run_epoch_imageonly(model, dl, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()

    if train:
        set_backbone_bn_eval(model.backbone)

    probs_all = []
    preds_all = []
    labels_all_epoch = []
    losses_all = []

    for x1, x2, h1, h2, y in dl:
        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)
        h1 = h1.to(DEVICE)
        h2 = h2.to(DEVICE)
        y = y.to(DEVICE)

        with torch.set_grad_enabled(train):
            out = model(x1, x2, h1, h2)
            loss = criterion(out, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        prob = torch.softmax(out, dim=1)[:, 1]
        pred = torch.argmax(out, dim=1)

        probs_all.append(prob.detach().cpu())
        preds_all.append(pred.detach().cpu())
        labels_all_epoch.append(y.detach().cpu())
        losses_all.append(loss.item())

    probs_all = torch.cat(probs_all).numpy()
    preds_all = torch.cat(preds_all).numpy()
    labels_all_epoch = torch.cat(labels_all_epoch).numpy()

    return float(np.mean(losses_all)), probs_all, preds_all, labels_all_epoch

# ======================================================================
# 7B.
# ======================================================================
def run_epoch_fusion(model, dl, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()

    if train:
        set_backbone_bn_eval(model.backbone)

    probs_all = []
    preds_all = []
    labels_all_epoch = []
    losses_all = []

    for x1, x2, h1, h2, struct1, struct2, y in dl:
        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)
        h1 = h1.to(DEVICE)
        h2 = h2.to(DEVICE)
        struct1 = struct1.to(DEVICE)
        struct2 = struct2.to(DEVICE)
        y = y.to(DEVICE)

        with torch.set_grad_enabled(train):
            out = model(x1, x2, h1, h2, struct1, struct2)
            loss = criterion(out, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        prob = torch.softmax(out, dim=1)[:, 1]
        pred = torch.argmax(out, dim=1)

        probs_all.append(prob.detach().cpu())
        preds_all.append(pred.detach().cpu())
        labels_all_epoch.append(y.detach().cpu())
        losses_all.append(loss.item())

    probs_all = torch.cat(probs_all).numpy()
    preds_all = torch.cat(preds_all).numpy()
    labels_all_epoch = torch.cat(labels_all_epoch).numpy()

    return float(np.mean(losses_all)), probs_all, preds_all, labels_all_epoch
