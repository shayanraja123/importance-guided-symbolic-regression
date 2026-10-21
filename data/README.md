# Data

`canonical/` holds the exact train/test splits used in the thesis experiments, one folder per dataset:

| File | Contents |
|---|---|
| `X_train.parquet`, `X_test.parquet` | Features, standardized with statistics from the training split only |
| `y_train.pkl`, `y_test.pkl` | Regression targets (pandas objects saved with `pd.to_pickle`) |
| `meta.json` | Sample counts and feature names |

All splits are 80/20 with `random_state=42`. The pickled targets can be read with `pd.read_pickle(...)`. As with any pickle, only load files from a source you trust.

## Sources

| Folder | Dataset | Source | Rows (train/test) | Features |
|---|---|---|---:|---:|
| `california_housing` | California Housing | [scikit-learn `fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) (StatLib) | 16,512 / 4,128 | 8 |
| `airfoil` | Airfoil Self-Noise | [UCI ML Repository](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) | 1,202 / 301 | 5 |
| `wine_red` | Wine Quality (red) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/186/wine+quality) | 1,279 / 320 | 11 |
| `wine_white` | Wine Quality (white) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/186/wine+quality) | 3,918 / 980 | 11 |
| `bike_hour` | Bike Sharing (hourly) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | 13,903 / 3,476 | 12 |
| `boston_housing` | Boston Housing | `hou_all.csv`, a common copy of the original Harrison & Rubinfeld (1978) data | 404 / 102 | 13 |
| `openml_586` | OpenML dataset 586 | [openml.org/d/586](https://www.openml.org/d/586) | 800 / 200 | 25 |
| `openml_589` | OpenML dataset 589 | [openml.org/d/589](https://www.openml.org/d/589) | 800 / 200 | 25 |
| `openml_607` | OpenML dataset 607 | [openml.org/d/607](https://www.openml.org/d/607) | 800 / 200 | 50 |
| `openml_616` | OpenML dataset 616 | [openml.org/d/616](https://www.openml.org/d/616) | 400 / 100 | 50 |
| `openml_618` | OpenML dataset 618 | [openml.org/d/618](https://www.openml.org/d/618) | 800 / 200 | 50 |
| `openml_620` | OpenML dataset 620 | [openml.org/d/620](https://www.openml.org/d/620) | 800 / 200 | 25 |
| `openml_637` | OpenML dataset 637 | [openml.org/d/637](https://www.openml.org/d/637) | 400 / 100 | 50 |

These processed files are redistributed for reproducibility. Each dataset remains under the terms of its original source; see the linked pages. `bike_hour` was run but is not part of the thesis results. The wine feature names contain literal quote characters (for example `"fixed acidity"`) carried over from the original CSV header.

## A note on Boston Housing

The Boston Housing data includes a variable (`black`) derived from the proportion of Black residents by town, based on assumptions that have been widely criticized. scikit-learn removed the dataset from its loaders in version 1.2 for this reason. It is included here only because it is a standard benchmark in the feature-generation literature the thesis compares against, and its results appear in the thesis. It should not be used to draw conclusions about housing or demographics.
