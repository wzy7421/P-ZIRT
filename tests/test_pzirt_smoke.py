from __future__ import annotations

import argparse

import numpy as np

import pzirt_model as pzirt


def tiny_args() -> argparse.Namespace:
    return argparse.Namespace(
        epochs=1,
        patience=1,
        batch_size=512,
        hidden_dim=16,
        group_dim=4,
        dropout=0.05,
        lr=1e-3,
        weight_decay=1e-5,
        group_l2=1e-6,
        grad_clip=5.0,
        cpu=True,
        verbose=False,
    )


def test_synthetic_demo_schema_and_sparsity() -> None:
    df = pzirt.make_synthetic_v2x(n=400, seed=7)
    assert {"road_lane", "decoding_rate", "packet_quality", "proxy"}.issubset(df.columns)
    assert df["proxy"].between(0, 1).all()
    assert (df["proxy"] == 0).mean() > 0.5


def test_prepare_data_group_split_and_baselines() -> None:
    df = pzirt.make_synthetic_v2x(n=600, seed=11)
    data = pzirt.prepare_data(
        df=df,
        target="proxy",
        group_col="road_lane",
        feature_cols=None,
        provenance_cols=["decoding_rate", "packet_quality"],
        weight_col=None,
        split="group",
        seed=11,
    )
    assert data.x_train.shape[1] == data.x_test.shape[1]
    assert data.n_groups > 1
    assert np.all((data.w_train >= 0.05) & (data.w_train <= 1.0))

    metrics = pzirt.baseline_metrics(data.y_test, data.y_train)
    assert "always_zero" in metrics
    assert "prevalence_prob_mean_value" in metrics
    assert metrics["always_zero"]["rmse"] >= 0


def test_train_pzirt_one_epoch_cpu_smoke() -> None:
    pzirt.set_seed(13)
    df = pzirt.make_synthetic_v2x(n=500, seed=13)
    data = pzirt.prepare_data(
        df=df,
        target="proxy",
        group_col="road_lane",
        feature_cols=None,
        provenance_cols=["decoding_rate", "packet_quality"],
        weight_col=None,
        split="group",
        seed=13,
    )
    model, metrics = pzirt.train_pzirt(data, tiny_args())
    assert model is not None
    assert metrics["rmse"] >= 0
    assert "pr_auc" in metrics
