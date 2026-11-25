# Cleaning a Real-World Kaggle Dataset — Missing Values & Outliers

Dataset: **Titanic** (Kaggle *Titanic – Machine Learning from Disaster*, `train.csv`, 891 rows).

## Task steps covered
| # | Step | Where / how |
|---|------|-------------|
| 1 | Download dataset and load into DataFrame | Notebook Step 1 (Kaggle CLI instructions included; copy bundled in `data/raw/`) |
| 2 | Analyze missing patterns, decide imputation | Notebook Step 2: missing chart + matrix, MCAR/MAR/MNAR checks, strategy table |
| 3 | Apply IQR method to detect outliers | Notebook Step 3: fences, report table, boxplots |
| 4 | Remove or cap outliers | Notebook Step 4: remove-vs-cap comparison, chosen strategy = cap |
| 5 | Export cleaned data | Notebook Step 5: validation asserts + `data/cleaned/titanic_cleaned.csv` |

## Cleaning decisions
| Issue | Action |
|---|---|
| `Age` – 177 missing (19.9%), depends on class (MAR) | Median by Title x Pclass |
| `Cabin` – 687 missing (77.1%), informative (MNAR) | Dropped; kept `Has_Cabin` flag |
| `Embarked` – 2 missing | Mode |
| `Fare` – 116 IQR outliers (13%), 90% are 1st class | Capped at upper fence (skew 4.79 -> 1.08) |
| `Age` – 22 IQR outliers (2.5%) | Capped at upper fence (~60.4) |

Capping was chosen over removal: removing would delete 13% of rows, almost all 1st-class passengers with ~2x average survival (selection bias).

## Structure
```
kaggle-data-cleaning/
├── data/raw/titanic.csv                  # original Kaggle file
├── data/cleaned/titanic_cleaned.csv      # final output (891 rows, 0 nulls)
├── data/cleaned/cleaning_log.csv         # what was changed and why
├── data/cleaned/outlier_report.csv       # IQR fences and counts
├── notebooks/kaggle_data_cleaning.ipynb  # main notebook (executed)
├── images/                               # 6 charts
├── clean_data.py                         # same pipeline as a CLI script
├── build_notebook.py                     # rebuilds the notebook
└── requirements.txt
```

## Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/kaggle_data_cleaning.ipynb     # Run All
# or, without Jupyter:
python clean_data.py --raw data/raw/titanic.csv --out data/cleaned/titanic_cleaned.csv
```
To use another Kaggle dataset, replace `data/raw/titanic.csv` and adapt the column names in Steps 2-4.
