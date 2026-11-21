# PEM-GeoCast Model

This repository releases the final PEM-GeoCast Compact-B model architecture
used in the reported experiments. It contains the core network, directly
required custom layers, the frozen architecture configuration, and a model
inspection utility.

## Release scope

Included:

- `pem_geocast.py`: final model architecture and custom network layers.
- `config/pem_geocast_compact_b.json`: frozen architecture configuration.
- `scripts/inspect_model.py`: architecture, parameter-count, checkpoint-loading,
  and forward-shape checks.
- `requirements.txt`: principal runtime dependency versions.

Not included:

- source or derived datasets;
- outbreak-reporting ASOF reconstruction;
- P/RN/U label construction;
- feature preprocessing or data loaders;
- geographic graph assets or the formal node order;
- training and evaluation pipelines;
- baseline models, bootstrap analyses, predictions, results, or checkpoints.

Consequently, this is a **model-architecture release**, not a complete
end-to-end reproduction package. The manuscript's predictions and performance
statistics cannot be reproduced from this repository alone.

## Frozen architecture

The model expects an input tensor with shape `[batch, nodes, 6, 155]`. The
first three channels are the disease-history channels; the remaining 152 are
local predictors. The final Compact-B forward graph uses:

- the full six-month temporal state;
- a disease-history encoder;
- a node-local persistence expert;
- an emergence expert over LOCAL, GEO1, GEO2, and GEO3;
- fixed equal fusion of the four active relations;
- disease and forecast-horizon embeddings; and
- a learned persistence/emergence conditional gate.

The adaptive graph and learned temporal/relation routing modules remain
instantiated but frozen. They are retained solely for strict state-dictionary
compatibility with the formal checkpoints and do not contribute to the final
Compact-B forward path.

The formal architecture has 255,932 trainable parameters and 526,532 total
instantiated parameters, including frozen compatibility modules.

## Environment

The formal experimental environment used Python 3.10.6, NumPy 1.23.5, and
PyTorch 1.12.1+cu113. The supplied requirements file specifies the corresponding
principal Python package versions; CUDA installation must be selected for the
target system separately.

## Architecture smoke test

The following command uses a small synthetic ring graph. It verifies that the
released model imports, has the expected trainable parameter count, and
completes a forward pass. It does **not** validate epidemiological performance
or reproduce a manuscript experiment.

```bash
python scripts/inspect_model.py --smoke-test
```

Expected output includes:

```text
"trainable_parameters": 255932
"logit_shape": [1, 8, 3, 3]
```

## Inspection with compatible private assets

If an authorized user has the frozen graph contract and checkpoint, the model
can be checked with:

```bash
python scripts/inspect_model.py \
  --graph-contract /path/to/graph_contract.npz \
  --checkpoint /path/to/checkpoint.pt
```

Only load checkpoints obtained from a trusted source. PyTorch checkpoint files
may contain serialized Python objects in addition to tensor weights.

The graph contract must contain:

- `geo_hop1_edge_index` with 14,854 directed edges;
- `geo_hop2_edge_index` with 30,970 directed edges; and
- `geo_hop3_edge_index` with 48,054 directed edges.

Each array must have shape `[2, E]` or `[E, 2]` and must use the same 3,236-node
order as the model input. Loading a checkpoint is strict: missing or unexpected
state-dictionary keys raise an error.

## Reproducibility boundary

Passing the smoke test demonstrates software-level model construction and
forward execution only. It does not establish reproduction of the paper's data
pipeline, trained models, predictions, statistical analyses, or conclusions.
