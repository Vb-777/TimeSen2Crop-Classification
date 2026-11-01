# Synthetic demo run (NOT real accuracy)

Produced by `python -m cropclf synth` + `benchmark --epochs 25` on generated data (5 tiles × 16 crops × 60 pixels). It only shows that the pipeline runs end to end; the real numbers come from the TimeSen2Crop data.

| Model | Test OA | Kappa | Macro-F1 | Weighted-F1 | Val OA | Best epoch | Train time |
|---|---|---|---|---|---|---|---|
| transformer | 59.90% | 0.572 | 59.48% | 59.48% | 66.15% | 17 | 21 s |
| tempcnn | 59.69% | 0.570 | 58.53% | 58.53% | 65.00% | 15 | 5 s |
| lstm | 57.60% | 0.548 | 56.79% | 56.79% | 67.50% | 24 | 9 s |
| weighted-lstm | 57.60% | 0.548 | 56.79% | 56.79% | 67.50% | 24 | 9 s |
| rf | 55.21% | 0.522 | 54.51% | 54.51% | 65.62% | – | 2 s |

Test pixels: 960 · validation pixels: 960

## Per-class F1 on the test tile

| Crop | Support | transformer | tempcnn | lstm | weighted-lstm | rf |
|---|---|---|---|---|---|---|
| Legumes | 60 | 0.58 | 0.55 | 0.53 | 0.53 | 0.39 |
| Grassland | 60 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Maize | 60 | 0.48 | 0.47 | 0.47 | 0.47 | 0.44 |
| Potato | 60 | 0.32 | 0.24 | 0.14 | 0.14 | 0.21 |
| Sunflower | 60 | 0.33 | 0.38 | 0.20 | 0.20 | 0.29 |
| Soy | 60 | 0.58 | 0.57 | 0.50 | 0.50 | 0.54 |
| Winter Barley | 60 | 0.31 | 0.18 | 0.38 | 0.38 | 0.26 |
| Winter Caraway | 60 | 0.95 | 0.85 | 0.89 | 0.89 | 0.82 |
| Rye | 60 | 0.32 | 0.46 | 0.43 | 0.43 | 0.35 |
| Rapeseed | 60 | 0.99 | 0.97 | 0.98 | 0.98 | 0.97 |
| Beet | 60 | 0.27 | 0.57 | 0.43 | 0.43 | 0.31 |
| Spring Cereals | 60 | 0.74 | 0.51 | 0.36 | 0.36 | 0.57 |
| Winter Wheat | 60 | 0.47 | 0.47 | 0.45 | 0.45 | 0.41 |
| Winter Triticale | 60 | 0.21 | 0.18 | 0.35 | 0.35 | 0.21 |
| Permanent Plantation | 60 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 |
| Other Crops | 60 | 0.97 | 0.98 | 0.99 | 0.99 | 0.98 |

Reference (Weikmann et al., IEEE JSTARS 2021, same tiles, 12 monthly composites): test OA TempCNN 81.71 %, LSTM 83.44 %, Transformer 84.44 %, weighted LSTM 85.39 %.
