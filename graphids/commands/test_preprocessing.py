"""Validate preprocessing: build dataset from raw data, check feature dims.

Usage:
    python -m graphids test-preprocessing [--dataset hcrl_ch]
"""

from __future__ import annotations

import argparse

from graphids.log import get_logger
import torch

log = get_logger(__name__)


def test_preprocessing(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate preprocessing pipeline")
    parser.add_argument("--dataset", default="hcrl_ch")
    args = parser.parse_args(argv)

    from graphids.core.preprocessing.datamodule import CANBusDataModule
    from graphids.core.preprocessing.features import N_EDGE_FEATURES, N_NODE_FEATURES

    dm = CANBusDataModule(dataset=args.dataset)
    dm.setup("fit")

    n_train = len(dm.train_dataset)
    n_val = len(dm.val_dataset)
    log.info("dataset_built", dataset=args.dataset, train=n_train, val=n_val)
    assert n_train > 0, "No training graphs produced"

    g = dm.train_dataset[0]
    log.info("sample_graph", x=list(g.x.shape), edge_index=list(g.edge_index.shape),
             edge_attr=list(g.edge_attr.shape), y=int(g.y))

    assert g.x.shape[1] == N_NODE_FEATURES, (
        f"Expected {N_NODE_FEATURES} node features, got {g.x.shape[1]}")
    assert g.edge_attr.shape[1] == N_EDGE_FEATURES, (
        f"Expected {N_EDGE_FEATURES} edge features, got {g.edge_attr.shape[1]}")
    assert not torch.isnan(g.x).any(), "NaN in node features"
    assert not torch.isnan(g.edge_attr).any(), "NaN in edge features"

    log.info("preprocessing_valid", dataset=args.dataset)


main = test_preprocessing
