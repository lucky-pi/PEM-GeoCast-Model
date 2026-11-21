"""Inspect the released PEM-GeoCast architecture without training a model.

This utility can run a synthetic architecture smoke test or instantiate the
formal 3236-node architecture from user-supplied geographic edge arrays. It
does not reproduce manuscript preprocessing, training, predictions, or metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pem_geocast import PEMGeoCast  # noqa: E402


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_edge_index(value: np.ndarray, name: str) -> np.ndarray:
    edge_index = np.asarray(value, dtype=np.int64)
    if edge_index.ndim != 2:
        raise ValueError(f"{name} must be rank 2, got {edge_index.shape}")
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T
    if edge_index.shape[0] != 2:
        raise ValueError(f"{name} must have shape [2,E], got {edge_index.shape}")
    return edge_index


def load_formal_edges(path: Path, config: dict) -> dict[int, np.ndarray]:
    contract = np.load(path, allow_pickle=False)
    keys = config["geographic_relations"]["required_npz_keys"]
    expected = config["geographic_relations"]["formal_directed_edge_counts"]
    edges = {}
    for hop, key in enumerate(keys, start=1):
        if key not in contract:
            raise KeyError(f"Missing required graph array: {key}")
        edge_index = normalize_edge_index(contract[key], key)
        if edge_index.shape[1] != expected[key]:
            raise ValueError(
                f"{key} has {edge_index.shape[1]} directed edges; "
                f"expected {expected[key]}"
            )
        edges[hop] = edge_index
    return edges


def synthetic_edges(n_nodes: int) -> dict[int, np.ndarray]:
    nodes = np.arange(n_nodes, dtype=np.int64)
    edges = {}
    for hop in (1, 2, 3):
        edges[hop] = np.stack([nodes, (nodes + hop) % n_nodes])
    return edges


def checkpoint_state(path: Path) -> dict:
    try:
        # Formal checkpoints contain training metadata in addition to tensors.
        # Use this option only with a checkpoint from a trusted source.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch 1.12 does not expose the weights_only argument.
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model_state", "state_dict", "model"):
            state = checkpoint.get(key)
            if isinstance(state, dict) and state:
                return state
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("No supported model state dictionary found in checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "pem_geocast_compact_b.json",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--graph-contract", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke_test:
        n_nodes = 8
        edges = synthetic_edges(n_nodes)
        adaptive_topk = 4
    else:
        n_nodes = config["input"]["formal_node_count"]
        edges = load_formal_edges(args.graph_contract, config)
        adaptive_topk = config["dimensions"]["adaptive_topk"]

    model = PEMGeoCast(
        input_dim=config["input"]["channels"],
        geo_edges=edges,
        n_nodes=n_nodes,
        hidden=config["dimensions"]["hidden"],
        embed_dim=config["dimensions"]["embedding"],
        adaptive_topk=adaptive_topk,
        dropout=config["dimensions"]["dropout"],
    )

    if args.checkpoint is not None:
        model.load_state_dict(checkpoint_state(args.checkpoint), strict=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    expected_trainable = config["parameter_counts"]["trainable"]
    if trainable != expected_trainable:
        raise RuntimeError(
            f"Trainable parameter mismatch: {trainable} != {expected_trainable}"
        )

    torch.manual_seed(20261121)
    model.eval()
    x = torch.randn(
        1,
        n_nodes,
        config["input"]["history_months"],
        config["input"]["channels"],
    )
    with torch.inference_mode():
        logits, diagnostics = model(x, return_diagnostics=True)

    report = {
        "mode": "synthetic_smoke_test" if args.smoke_test else "formal_shape_check",
        "checkpoint_loaded": args.checkpoint is not None,
        "nodes": n_nodes,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "logit_shape": list(logits.shape),
        "diagnostic_shapes": {
            name: list(value.shape) for name, value in diagnostics.items()
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
