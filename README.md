# AlphaEarth Poverty Prediction

Seven-country H3 resolution-7 pipeline using AlphaEarth, other geospatial features, and DHS labels to predict wealth, sanitation, women's education, mobile ownership, and stunting.

This folder has the three main notebooks, the V1 report, and the combined metrics from the current runs.

## Project status

- Built the H3 spine and the full merged feature table.
- Added GEE features, local population/building/road features, and five DHS outcomes.
- Ran blocked spatial CV, leave-one-country-out tests, XGBoost, GraphSAGE, and a self-supervised embedding.
- Wrote the V1 report and collected the current metrics.

This is still work in progress: seven countries, imperfect source-year matching, and some important health, access, institutional, and shock variables are still missing.

## Folder contents

| File | Purpose |
|---|---|
| `h3_gee_tabular_features_colab.ipynb` | Creates the H3 spine and extracts the Google Earth Engine feature blocks: AlphaEarth, VIIRS, MODIS NDVI, and GHSL building height. It also constructs Google Open Buildings morphology for five countries. |
| `h3_local_worldpop_osm_features_colab.ipynb` | Creates DHS labels, WorldPop demographic features, OSM road features, RWI audit fields, and local Google Open Buildings features for Nigeria and Tanzania, then performs the final one-to-one H3 merge. |
| `h3_strict_spatial_training_colab_final.ipynb` | Selects model features, creates strict spatial folds, trains XGBoost, self-supervised, and GraphSAGE models, runs LOCO evaluation, and exports metrics and predictions. |
| `alphaearth_augmented_features_V1.pdf` | Preliminary V1 methodology, results, interpretation, limitations, and related-work report. |
| `all_metrics_with GNNsingleheadtraining_smoothing.csv` | 410 fold-level or held-out-country metric records from the final model runs. |

## End-to-end workflow

```text
Country boundaries
    |
    v
H3 resolution-7 country spine
    |
    +--> Google Earth Engine
    |      - AlphaEarth
    |      - VIIRS nightlights
    |      - MODIS NDVI
    |      - GHSL building height
    |
    +--> Local processing
           - WorldPop
           - OpenStreetMap roads
           - Google Open Buildings
           - DHS outcomes
           - RWI audit fields
    |
    v
cells_full_tabular_v2.parquet
    |
    +--> Blocked spatial cross-validation
    +--> Leave-one-country-out evaluation
    |
    +--> XGBoost
    +--> GraphSAGE
    +--> Self-supervised fused embedding + Ridge probe
```

## Countries and sample coverage

The upstream GEE spine was created for nine countries and contained 904,494 H3 resolution-7 cells. Ethiopia and Togo were removed before modeling because the available inputs were not consistent with the complete five-outcome analysis: Ethiopia used an older SPA GPS source and Togo lacked the required anthropometry records.

The active modeling table contains 657,863 H3 cells across seven countries.

| Country | ISO3 | H3 cells | Wealth | Sanitation | Women's education | Stunting | Mobile |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ghana | GHA | 60,433 | 602 | 602 | 602 | 600 | 602 |
| Gambia | GMB | 2,316 | 172 | 172 | 172 | 172 | 172 |
| Kenya | KEN | 106,744 | 1,490 | 1,490 | 1,490 | 1,489 | 1,490 |
| Nigeria | NGA | 190,814 | 1,350 | 1,350 | 1,350 | 1,313 | 1,350 |
| Sierra Leone | SLE | 17,349 | 464 | 464 | 464 | 454 | 464 |
| Tanzania | TZA | 158,287 | 596 | 596 | 596 | 592 | 596 |
| Zambia | ZMB | 121,920 | 512 | 512 | 512 | 512 | 512 |
| **Total** |  | **657,863** | **5,186** | **5,186** | **5,186** | **5,132** | **5,186** |

After applying the 5 km boundary buffer, 4,266 labeled cells are used for the four main outcomes and 4,219 for stunting.

## Main datasets

**GEE:** AlphaEarth annual embeddings, VIIRS monthly nightlights, MODIS Terra/Aqua NDVI, GHSL building height.

**Local/other:** WorldPop population and age-sex rasters, OpenStreetMap roads, Google Open Buildings, DHS recode and GPS files, Meta RWI audit fields.

Raw data and the final merged parquet are not included in this folder. DHS recode and GPS files require the appropriate DHS access permissions.

## Feature construction

- **Spatial unit:** one row per H3 resolution-7 cell, GAUL country/Admin-1 boundaries, Admin-1 assigned by centroid, GEE values summarized over the full H3 polygon, tables joined one-to-one on country and H3 index.
- **AlphaEarth:** 64 annual embedding dimensions, polygon means, 30 m reduction scale, chunked exports, years capped at 2024.
- **Nightlights:** 24 months of VIIRS, mean radiance, raw volatility, log trend, log volatility, low-coverage months masked.
- **NDVI:** MODIS Terra and Aqua, 23 16-day periods, annual mean, p90-p10 seasonal amplitude, peak day of year.
- **WorldPop:** population count and density, youth and elderly dependency ratios, sex ratio, population-gravity fields computed but dropped from reported models.
- **Buildings:** GHSL mean/p90/max height, Open Buildings count, density, footprint share, area statistics, size Gini, spacing regularity, orientation entropy.
- **Roads:** OSM road density and class, segment/node/intersection counts, dead-end ratio, degree shares, beta connectivity, bearing entropy, midpoint assignment used for segments.

**DHS outcomes:** wealth (`hv271`, index), sanitation (`hv205`, improved share), women's education (`v133`, years), stunting (`hw70`, HAZ below -2), mobile ownership (`hv243a`, ownership share). Survey weights are applied within clusters, GPS points are snapped to H3, and clusters sharing a cell are averaged.

## Final model feature matrix

The merged table has 145 columns before exclusions and 101 final model features: 64 AlphaEarth dimensions, 37 tabular features.

**Tabular groups:** demographics (5), nightlight temporal (3), nightlight amount (1), NDVI temporal (2), NDVI amount (1), building structure (7), building amount (6), road topology (7), road amount (5).

**Feature split:** orthogonal/context (24), amount/intensity (13).

**Evaluated feature sets:** `ae_only` (64), `tab_only` (37), `ae_plus_amount` (77), `ae_plus_orthogonal` (88), `full` (101).

The training code excludes DHS columns, RWI fields, coordinates, area, source/effective-year fields, quality flags, accessibility features, and selected QA/composite columns.

## Training and evaluation

### Strict spatial cross-validation

- Five folds, country-prefixed 1-degree blocks, balanced by labeled cells/country/total cells.
- A 5 km boundary buffer is excluded from training and testing, coordinates are dropped, preprocessing is fitted on training cells only.
- Results are pooled across all out-of-fold predictions.

### Leave-one-country-out transfer

- One country is held out at a time and excluded from supervised and self-supervised fitting.
- Results are pooled across held-out predictions; per-country R2 is mainly diagnostic for smaller countries and weaker outcomes.

### Metrics

R2, RMSE, MAE, Spearman rank correlation.

## Models

### XGBoost

XGBoost is the main supervised tabular baseline. It is evaluated on all five feature sets under spatial CV and LOCO, with GPU use when available and CPU fallback.

### Self-supervised fused embedding

Separate AlphaEarth and tabular encoders are fused into a reusable representation, `Z`, using reconstruction, cross-view alignment, feature masking, and variance regularization. DHS outcomes are not used to train the representation; a frozen Ridge probe is fitted afterward.

### GraphSAGE

The final graph model trains one GraphSAGE model per outcome over H3 adjacency, with block-level validation and early stopping. Train and test graphs are separate so message passing cannot cross the held-out boundary. GraphSAGE is evaluated for `ae_only` and `full` under spatial CV; LOCO GraphSAGE is not in the current notebook.

## Main results

All values below are pooled R2 from the final prediction tables.

### XGBoost feature-set comparison

#### Spatial cross-validation

| Outcome | AE only | Tabular only | AE + amount | AE + orthogonal | Full |
|---|---:|---:|---:|---:|---:|
| Wealth | 0.635 | 0.736 | 0.743 | 0.718 | **0.750** |
| Sanitation | 0.424 | 0.503 | 0.514 | 0.496 | **0.526** |
| Women's education | 0.663 | 0.690 | 0.707 | 0.703 | **0.723** |
| Mobile ownership | 0.468 | 0.501 | 0.504 | 0.498 | **0.516** |
| Stunting | 0.203 | 0.187 | 0.207 | **0.216** | **0.216** |

#### Leave-one-country-out

| Outcome | AE only | Tabular only | AE + amount | AE + orthogonal | Full |
|---|---:|---:|---:|---:|---:|
| Wealth | 0.505 | 0.657 | **0.682** | 0.643 | 0.673 |
| Sanitation | 0.281 | 0.385 | **0.415** | 0.387 | 0.410 |
| Women's education | 0.171 | **0.329** | 0.275 | 0.311 | 0.306 |
| Mobile ownership | 0.004 | 0.097 | 0.106 | 0.085 | **0.126** |
| Stunting | -0.070 | -0.042 | -0.030 | **-0.008** | -0.016 |

### Model comparison

#### Spatial cross-validation R2

| Outcome | XGB full | GNN AE only | GNN full | Frozen SSL `Z` |
|---|---:|---:|---:|---:|
| Wealth | 0.750 | 0.649 | **0.751** | 0.706 |
| Sanitation | **0.526** | 0.422 | 0.513 | 0.467 |
| Women's education | **0.723** | 0.659 | 0.706 | 0.546 |
| Mobile ownership | 0.516 | 0.384 | **0.541** | 0.318 |
| Stunting | **0.216** | **0.216** | 0.204 | 0.160 |

#### Leave-one-country-out

| Outcome | XGB full R2 | XGB Spearman | SSL `Z` R2 | SSL `Z` Spearman |
|---|---:|---:|---:|---:|
| Wealth | 0.673 | 0.832 | 0.635 | 0.811 |
| Sanitation | 0.410 | 0.647 | 0.380 | 0.637 |
| Women's education | 0.306 | 0.530 | 0.184 | 0.459 |
| Mobile ownership | 0.126 | 0.478 | -0.119 | 0.317 |
| Stunting | -0.016 | 0.236 | -0.095 | 0.056 |

## Main takeaways

1. **Wealth is the strongest and most transferable outcome.** The full XGBoost model reaches 0.750 spatial-CV R2 and 0.673 pooled LOCO R2. The result suggests that settlement intensity, built form, roads, lighting, population, and the AlphaEarth representation jointly carry substantial material-welfare information.

2. **Engineered tabular layers add meaningful information beyond AlphaEarth.** The full stack improves over AlphaEarth-only for wealth, sanitation, education, and mobile ownership. The improvement is especially important in LOCO, indicating that the engineered variables contain signal that transfers across countries.

3. **Augmentation appears to be a larger lever than graph message passing.** The full GraphSAGE model is effectively tied with XGBoost for wealth and performs better for mobile ownership, but it is slightly lower for sanitation, education, and stunting. Much of the usable spatial information appears to be present in the per-cell feature matrix.

4. **The self-supervised representation is promising for wealth and sanitation.** The frozen embedding trained without DHS labels reaches 0.706/0.635 R2 for wealth and 0.467/0.380 for sanitation under spatial CV/LOCO. It retains much of the supervised signal but is not yet a general replacement for the full supervised models.

5. **Women's education is strongly spatially organized despite not being directly visible.** High spatial-CV performance likely reflects correlated long-run development, infrastructure, urbanization, and demographic gradients. Cross-country calibration remains weaker.

6. **Mobile ownership is locally predictable but transfer-sensitive.** Spatial models reach approximately 0.52-0.54 R2, while pooled LOCO is only 0.126 for full XGBoost. Country-specific telecom markets, pricing, rollout, and survey timing likely matter.

7. **Stunting is the current boundary case.** Performance remains near 0.20 in spatial CV and near zero or negative in LOCO. The current feature stack does not directly represent many likely drivers of child nutrition, including disease ecology, food security, health-service access, shocks, and survey season.

## Run order

### Prepare the environment

Use Python with Earth Engine/geospatial packages, scikit-learn, XGBoost, and PyTorch. The first notebook needs Earth Engine and Google Drive; the second needs the local raster, OSM, buildings, RWI, and authorized DHS files.

1. Run `h3_gee_tabular_features_colab.ipynb` to build the H3/GEE tables, about 7 main parquet outputs.
2. Run `h3_local_worldpop_osm_features_colab.ipynb` to build local features and labels, about 5 supporting files plus the final merged parquet.
3. Put the merged table at `h3_tabular_v2/cells_full_tabular_v2.parquet`, then run `h3_strict_spatial_training_colab_final.ipynb`; each run writes one timestamped folder with 13 fold, metric, and prediction files.

## Interpreting the included metrics CSV

`all_metrics_with GNNsingleheadtraining_smoothing.csv` has one record per spatial fold or held-out country, with the setup, sample size, and metrics.
Headline results use pooled prediction rows, not a simple mean of fold-level or country-level R2.

## Next to do

- Look at more countries, probably outside Africa too, and see whether any of this holds across continents.
- Run the 0/5/10 km buffer checks, clean up source-year matching, and separate LOCO ranking from calibration.
- Add the missing access/health/context variables: travel time, health services, food security, disease, institutions, shocks, and survey season.
- Do the model follow-ups: SHAP and residual maps, direct imagery comparisons, better SSL objectives, fuzzy DHS labels, and GraphSAGE LOCO.
