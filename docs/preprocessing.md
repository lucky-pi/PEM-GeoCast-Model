# Frozen input preprocessing

This document records the feature transformations used by the frozen
Compact-B training loader. It does not include source datasets, transformed
feature arrays, node-level records, or the private data loader.

## Runtime tensor layout

The model input has shape `[batch, nodes, history, channels]`, with 155 channels
in this order:

1. Disease-history indicators: ASF, FMD, PPR (3 channels).
2. LOCAL covariates (150 channels): 98 ERA5, 8 host-exposure, 24 host-data
   indicators, and 20 NDVI/quality channels.
3. Calendar encoding: month sine and cosine (2 channels).

`config/runtime_input_channels.csv` maps runtime channel indices to the
feature-manifest indices and names.

## Continuous-feature transform

For each continuous feature, the frozen loader applies TRAIN-derived
statistics. Let `x` be the input, `q01` and `q99` the TRAIN 1st and 99th
percentiles, `m` the TRAIN imputation statistic, `c` the TRAIN center, and `s`
the TRAIN robust scale. When both percentile bounds are finite and
`q99 >= q01`, values are clipped to `[q01, q99]`; non-finite values are then
replaced by `m`; finally the result is `(x - c) / max(abs(s), EPS)`. The cached
metadata records that preprocessing statistics were reused from training and
were not refit for evaluation.

This transform is used for the 98 ERA5 channels, 8 host-exposure channels, and
19 continuous NDVI/quality channels. Host statistics are indexed by host
species and feature.

## Indicator and deterministic channels

- The 24 host-data indicators are cast directly to `float32`; they bypass
  clipping, imputation, centering, and scaling.
- `ndvi_lag1_missing` is an identity indicator: non-finite values are replaced
  with zero, with no clipping or scaling.
- The three disease-history channels are the forecast-time-observable event
  history supplied by the frozen index. They are not imputed or scaled here.
- Calendar sine/cosine channels are deterministic and are not imputed or
  scaled.

## Scope and availability boundary

The channel manifest identifies the sources and the frozen causal-lag rule
recorded for each predictor. This release does not reconstruct historical
product vintages for ERA5, host statistics, or NDVI; it must not be described
as a complete historical real-time data-vintage replay. The transformations
above document preprocessing, not source-data release-time availability.

## Source provenance

The transformation details were checked against the final training-loader
source used in the local project audit (`stage3b_run_temporal_gru_tcn_baselines_v1.py`).
That full loader is intentionally not published because it includes private
data-pipeline and experiment code. Its audited SHA-256 is recorded in the
project's local provenance audit as
`e271ac2b720b7e58de2abd1d9324d1c755f7f2382aace30fb18b72bb9db291f6`. This
public release does not claim that the entire loader can be reconstructed from
this document alone.
