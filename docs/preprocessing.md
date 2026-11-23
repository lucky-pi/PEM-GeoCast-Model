# Frozen input preprocessing specification

This document follows the preprocessing description in the submitted manuscript and Supplementary Data S1. The field-level feature names, dictionary indices, runtime indices, source labels, semantic units, temporal and spatial rules, and preprocessing categories are given in [`../config/runtime_input_channels.csv`](../config/runtime_input_channels.csv).

## Runtime tensor layout

Each sample contains six consecutive historical months and 3,236 ADM1-equivalent nodes. Before the model forward pass, the loader converts `[B, T, N, F]` to `[B, N, T, F]` without changing channel values or temporal order. The 155 input channels are ordered as follows:

- 0–2: ASOF outbreak-history indicators for ASF, FMD, and PPR;
- 3–100: 98 ERA5 climate and derived features;
- 101–132: 32 host-related features, comprising 8 numerical exposures and 24 indicators;
- 133–152: 20 MODIS MOD13C2.061 vegetation features;
- 153–154: month sine and cosine encodings.

## Training-derived preprocessing

All data-dependent preprocessing statistics are estimated from TRAIN and applied unchanged to VALID, the internal temporal holdout, and the extended temporal sensitivity period. No evaluation-period observations are used to refit these parameters.

For the 98 ERA5 features and 12 continuous NDVI-derived features, the manuscript specifies field-appropriate clipping at frozen TRAIN 1st- and 99th-percentile limits, replacement of non-finite values using TRAIN imputation statistics, and robust scaling with a numerical scale floor. The 8 numerical host-exposure variables are log1p-transformed before their corresponding TRAIN-derived preprocessing. The 3 NDVI valid-observation counts retain count semantics, and the 4 vegetation-area fractions retain their proportion semantics in [0, 1].

The 24 host-source/missingness indicators and `ndvi_lag1_missing` remain binary indicators without ordinary continuous-variable scaling. The three ASOF outbreak-history channels and two cyclic calendar encodings are constructed directly.

## Forecast-time availability boundary

The reporting-date filter constrains outbreak-history predictors. ERA5 features use reference months through the forecast origin; NDVI uses the month before the origin and earlier; FAOSTAT uses the prior reference year. Host spatial allocation and administrative boundaries are frozen retrospectively. Historical publication-time vintages for ERA5, MODIS, GLW, FAOSTAT, and boundary products are not reconstructed, so this specification does not claim a complete real-time historical product-vintage replay.

## Reproducibility scope

This page records the method as stated in the manuscript and Supplementary Data S1. The model-only repository does not include raw data, feature arrays, fitted TRAIN statistics, or the full private data loader. The CSV is the submitted field-level manifest; it is not a substitute for those assets.
