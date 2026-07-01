#!/usr/bin/env python3
"""
P-ZIRT: Provenance-aware Zero-Inflated Road-lane tabular neural prototype.

This file is a compact research prototype for zero-inflated roadside V2X proxy
monitoring. It supports:
  - a zero-inflated Beta likelihood for continuous targets in [0, 1];
  - provenance/reliability weighting for partially decoded data;
  - road-lane group embeddings for deployment diagnostics;
  - calibrated nonzero probabilities and TRC-style evaluation metrics.

Example demo:
  python pzirt_model.py

Example CSV:
  python pzirt_model.py --csv data.csv --target proxy \
    --group-col road_lane --provenance-cols decoding_rate packet_quality
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


EPS = 1e-6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    g_train: np.ndarray
    w_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    g_val: np.ndarray
    w_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    g_test: np.ndarray
    w_test: np.ndarray
    n_groups: int
    feature_names: list[str]


class RoadLaneDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, g: np.ndarray, w: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.g = torch.tensor(g, dtype=torch.long)
        self.w = torch.tensor(w, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.g[idx], self.w[idx]


class PZIRT(nn.Module):
    """
    Zero-inflated continuous proxy model.

    The model estimates:
      pi(x): P(y > 0 | x), the nonzero-event probability
      mu(x): E[y | y > 0, x], the nonzero target mean
      phi(x): Beta precision for nonzero values

    Expected proxy prediction is E[y | x] = pi(x) * mu(x).
    """

    def __init__(
        self,
        n_features: int,
        n_groups: int,
        hidden_dim: int = 128,
        group_dim: int = 16,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.group_embedding = nn.Embedding(max(n_groups, 1), group_dim)
        self.backbone = nn.Sequential(
            nn.Linear(n_features + group_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pi_head = nn.Linear(hidden_dim, 1)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.phi_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, group_id: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = self.group_embedding(group_id)
        h = self.backbone(torch.cat([x, emb], dim=1))
        pi = torch.sigmoid(self.pi_head(h)).squeeze(1).clamp(EPS, 1.0 - EPS)
        mu = torch.sigmoid(self.mu_head(h)).squeeze(1).clamp(EPS, 1.0 - EPS)
        phi = F.softplus(self.phi_head(h)).squeeze(1) + 2.0
        return {"pi": pi, "mu": mu, "phi": phi, "expected": pi * mu}


def zero_inflated_beta_nll(
    y: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    phi: torch.Tensor,
    reliability_weight: torch.Tensor,
) -> torch.Tensor:
    is_nonzero = (y > EPS).float()
    y_clip = y.clamp(EPS, 1.0 - EPS)
    alpha = (mu * phi).clamp_min(EPS)
    beta = ((1.0 - mu) * phi).clamp_min(EPS)
    beta_dist = torch.distributions.Beta(alpha, beta)

    zero_nll = -torch.log(1.0 - pi)
    nonzero_nll = -torch.log(pi) - beta_dist.log_prob(y_clip)
    nll = torch.where(is_nonzero > 0.5, nonzero_nll, zero_nll)
    return (nll * reliability_weight).mean()


class PlattCalibrator:
    """Simple sigmoid calibration for nonzero probabilities."""

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray, max_iter: int = 300) -> None:
        labels = labels.astype(np.float32)
        if len(np.unique(labels)) < 2:
            return
        logits = np.log(np.clip(probs, EPS, 1.0 - EPS) / np.clip(1.0 - probs, EPS, 1.0))
        x = torch.tensor(logits, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        a = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(a * x + b, y)
            loss.backward()
            return loss

        opt.step(closure)
        self.a = float(a.detach())
        self.b = float(b.detach())
        self.fitted = True

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.clip(probs, EPS, 1.0 - EPS)
        logits = np.log(np.clip(probs, EPS, 1.0 - EPS) / np.clip(1.0 - probs, EPS, 1.0))
        return 1.0 / (1.0 + np.exp(-(self.a * logits + self.b)))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    labels = labels.astype(float)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += mask.mean() * abs(conf - acc)
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, p_nonzero: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = (y_true > EPS).astype(int)
    prevalence = max(labels.mean(), EPS)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if labels.sum() > 0:
        nonzero_rmse = float(np.sqrt(np.mean((y_true[labels == 1] - y_pred[labels == 1]) ** 2)))
        nonzero_mae = float(np.mean(np.abs(y_true[labels == 1] - y_pred[labels == 1])))
    else:
        nonzero_rmse = float("nan")
        nonzero_mae = float("nan")

    if len(np.unique(labels)) > 1:
        pr_auc = float(average_precision_score(labels, p_nonzero))
        brier = float(brier_score_loss(labels, p_nonzero))
        base_brier = float(brier_score_loss(labels, np.full_like(labels, prevalence, dtype=float)))
        brier_skill = float(1.0 - brier / max(base_brier, EPS))
        ece = expected_calibration_error(p_nonzero, labels)
    else:
        pr_auc = float("nan")
        brier = float("nan")
        brier_skill = float("nan")
        ece = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "nonzero_rmse": nonzero_rmse,
        "nonzero_mae": nonzero_mae,
        "prevalence": float(prevalence),
        "pr_auc": pr_auc,
        "pr_lift": float(pr_auc / prevalence) if not math.isnan(pr_auc) else float("nan"),
        "brier": brier,
        "brier_skill": brier_skill,
        "ece": ece,
    }


def make_synthetic_v2x(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    road = rng.integers(0, 24, size=n)
    lane = rng.integers(0, 4, size=n)
    hour = rng.integers(0, 24, size=n)
    speed = rng.normal(45, 12, size=n).clip(3, 90)
    density = rng.gamma(2.0, 0.35, size=n).clip(0, 4)
    decoding_rate = rng.beta(8, 2, size=n)
    packet_quality = rng.beta(6, 3, size=n)
    source_type = rng.choice(["rsu_a", "rsu_b", "rsu_c"], size=n, p=[0.45, 0.35, 0.20])
    group_effect = (road % 5) * 0.12 + lane * 0.08
    rush = ((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))
    logit_nonzero = (
        -4.2
        + 0.55 * density
        - 0.025 * speed
        + 0.85 * rush.astype(float)
        + 0.35 * group_effect
        + 0.9 * (1.0 - decoding_rate)
    )
    p = 1.0 / (1.0 + np.exp(-logit_nonzero))
    nonzero = rng.binomial(1, p)
    severity_mean = 1.0 / (1.0 + np.exp(-(-2.0 + 0.7 * density + 0.5 * group_effect)))
    proxy = np.where(nonzero == 1, rng.beta(2 + 8 * severity_mean, 8), 0.0)

    return pd.DataFrame(
        {
            "road": road,
            "lane": lane,
            "road_lane": [f"r{r}_l{l}" for r, l in zip(road, lane)],
            "hour": hour,
            "speed": speed,
            "density": density,
            "decoding_rate": decoding_rate,
            "packet_quality": packet_quality,
            "source_type": source_type,
            "proxy": proxy.clip(0, 1),
        }
    )


def infer_feature_columns(
    df: pd.DataFrame,
    target: str,
    group_col: str | None,
    weight_col: str | None,
    explicit_features: Iterable[str] | None,
) -> list[str]:
    if explicit_features:
        return list(explicit_features)
    excluded = {target}
    if group_col:
        excluded.add(group_col)
    if weight_col:
        excluded.add(weight_col)
    return [c for c in df.columns if c not in excluded]


def reliability_from_provenance(
    df: pd.DataFrame,
    provenance_cols: list[str] | None,
    weight_col: str | None,
) -> np.ndarray:
    if weight_col and weight_col in df.columns:
        return df[weight_col].astype(float).to_numpy().clip(0.05, 1.0)
    if not provenance_cols:
        return np.ones(len(df), dtype=np.float32)
    cols = [c for c in provenance_cols if c in df.columns]
    if not cols:
        return np.ones(len(df), dtype=np.float32)
    arr = df[cols].apply(pd.to_numeric, errors="coerce").fillna(df[cols].median(numeric_only=True))
    values = arr.to_numpy(dtype=float)
    lo = np.nanmin(values, axis=0)
    hi = np.nanmax(values, axis=0)
    scaled = (values - lo) / np.maximum(hi - lo, EPS)
    return np.nanmean(scaled, axis=1).clip(0.05, 1.0).astype(np.float32)


def prepare_data(
    df: pd.DataFrame,
    target: str,
    group_col: str | None,
    feature_cols: list[str] | None,
    provenance_cols: list[str] | None,
    weight_col: str | None,
    split: str,
    seed: int,
) -> SplitData:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    df[target] = df[target].clip(0.0, 1.0)

    if group_col and group_col in df.columns:
        group_codes, _ = pd.factorize(df[group_col].astype(str), sort=True)
    else:
        group_codes = np.zeros(len(df), dtype=int)

    features = infer_feature_columns(df, target, group_col, weight_col, feature_cols)
    x_df = pd.get_dummies(df[features], dummy_na=True)
    x_df = x_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df[target].to_numpy(dtype=np.float32)
    w = reliability_from_provenance(df, provenance_cols, weight_col)
    g = group_codes.astype(np.int64)

    idx = np.arange(len(df))
    if split == "group" and len(np.unique(g)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_val_idx, test_idx = next(splitter.split(idx, groups=g))
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 1)
        train_idx_rel, val_idx_rel = next(splitter2.split(train_val_idx, groups=g[train_val_idx]))
        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]
    else:
        train_val_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=seed + 1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_df.iloc[train_idx].to_numpy(dtype=np.float32))
    x_val = scaler.transform(x_df.iloc[val_idx].to_numpy(dtype=np.float32))
    x_test = scaler.transform(x_df.iloc[test_idx].to_numpy(dtype=np.float32))

    return SplitData(
        x_train=x_train,
        y_train=y[train_idx],
        g_train=g[train_idx],
        w_train=w[train_idx],
        x_val=x_val,
        y_val=y[val_idx],
        g_val=g[val_idx],
        w_val=w[val_idx],
        x_test=x_test,
        y_test=y[test_idx],
        g_test=g[test_idx],
        w_test=w[test_idx],
        n_groups=int(g.max() + 1),
        feature_names=list(x_df.columns),
    )


def predict(model: PZIRT, x: np.ndarray, g: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    probs = []
    loader = DataLoader(
        RoadLaneDataset(x, np.zeros(len(x)), g, np.ones(len(x))),
        batch_size=1024,
        shuffle=False,
    )
    with torch.no_grad():
        for bx, _, bg, _ in loader:
            out = model(bx.to(device), bg.to(device))
            preds.append(out["expected"].cpu().numpy())
            probs.append(out["pi"].cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def train_pzirt(data: SplitData, args: argparse.Namespace) -> tuple[PZIRT, dict[str, float]]:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = PZIRT(
        n_features=data.x_train.shape[1],
        n_groups=data.n_groups,
        hidden_dim=args.hidden_dim,
        group_dim=args.group_dim,
        dropout=args.dropout,
    ).to(device)

    train_loader = DataLoader(
        RoadLaneDataset(data.x_train, data.y_train, data.g_train, data.w_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for bx, by, bg, bw in train_loader:
            bx, by, bg, bw = bx.to(device), by.to(device), bg.to(device), bw.to(device)
            out = model(bx, bg)
            nll = zero_inflated_beta_nll(by, out["pi"], out["mu"], out["phi"], bw)
            group_penalty = args.group_l2 * model.group_embedding.weight.pow(2).mean()
            loss = nll + group_penalty
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_pred, val_prob = predict(model, data.x_val, data.g_val, device)
        val_metrics = compute_metrics(data.y_val, val_pred, val_prob)
        val_score = val_metrics["rmse"] + val_metrics["brier"] if not math.isnan(val_metrics["brier"]) else val_metrics["rmse"]
        if val_score < best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if args.verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"epoch={epoch:03d} loss={np.mean(losses):.4f} "
                f"val_rmse={val_metrics['rmse']:.4f} val_brier={val_metrics['brier']:.4f}"
            )
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred, val_prob = predict(model, data.x_val, data.g_val, device)
    calibrator = PlattCalibrator()
    calibrator.fit(val_prob, (data.y_val > EPS).astype(int))
    test_pred, test_prob = predict(model, data.x_test, data.g_test, device)
    test_prob_cal = calibrator.transform(test_prob)

    metrics = compute_metrics(data.y_test, test_pred, test_prob_cal)
    return model, metrics


def baseline_metrics(y_test: np.ndarray, y_train: np.ndarray) -> dict[str, dict[str, float]]:
    labels_train = (y_train > EPS).astype(int)
    prevalence = float(labels_train.mean())
    historical_mean = float(y_train.mean())
    return {
        "always_zero": compute_metrics(y_test, np.zeros_like(y_test), np.zeros_like(y_test)),
        "prevalence_prob_mean_value": compute_metrics(
            y_test,
            np.full_like(y_test, historical_mean, dtype=float),
            np.full_like(y_test, prevalence, dtype=float),
        ),
    }


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    ordered = [
        "rmse",
        "mae",
        "nonzero_rmse",
        "nonzero_mae",
        "prevalence",
        "pr_auc",
        "pr_lift",
        "brier",
        "brier_skill",
        "ece",
    ]
    values = " | ".join(f"{k}={metrics[k]:.4f}" for k in ordered if k in metrics)
    print(f"{name}: {values}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P-ZIRT for zero-inflated V2X proxy monitoring.")
    parser.add_argument("--csv", type=str, default=None, help="Input CSV. If omitted, a synthetic demo is used.")
    parser.add_argument("--target", type=str, default="proxy", help="Target column in [0, 1].")
    parser.add_argument("--group-col", type=str, default="road_lane", help="Road-lane/site group column.")
    parser.add_argument("--feature-cols", nargs="*", default=None, help="Optional explicit feature columns.")
    parser.add_argument("--provenance-cols", nargs="*", default=["decoding_rate", "packet_quality"])
    parser.add_argument("--weight-col", type=str, default=None, help="Optional precomputed reliability weight.")
    parser.add_argument("--split", choices=["random", "group"], default="group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--group-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--group-l2", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        print("No CSV supplied. Running synthetic roadside V2X demo.")
        df = make_synthetic_v2x(seed=args.seed)

    data = prepare_data(
        df=df,
        target=args.target,
        group_col=args.group_col,
        feature_cols=args.feature_cols,
        provenance_cols=args.provenance_cols,
        weight_col=args.weight_col,
        split=args.split,
        seed=args.seed,
    )

    print(f"samples: train={len(data.y_train)} val={len(data.y_val)} test={len(data.y_test)}")
    print(f"features={data.x_train.shape[1]} groups={data.n_groups} split={args.split}")

    for name, metrics in baseline_metrics(data.y_test, data.y_train).items():
        print_metrics(name, metrics)

    _, metrics = train_pzirt(data, args)
    print_metrics("P-ZIRT", metrics)


if __name__ == "__main__":
    main()
