# Predicting Earthquake Damage Classification Models

Classification models for predicting severe building damage in the 2015 Nepal (Gorkha) earthquake, using the Nepal Earthquake Damage Assessment survey data.

**Status:** Lesson 4.1 (SQLite/DuckDB), Lesson 4.2 (Logistic Regression), and Lesson 4.3 (Decision Trees) complete.

## What this project does

- Loads the Nepal earthquake damage survey from CSV files using DuckDB
- Explores the relational database structure (4 tables, 234,835 buildings, 249,932 households)
- Builds a reusable data pipeline (`wrangle_nepal_data()`) that joins building features with damage grades and creates a binary `severe_damage` target
- Filters to the Gorkha district (epicenter of the earthquake)
- Establishes the baseline classification problem: predict severe damage (Grade 4–5) vs. non-severe (Grade 1–3)

## Data

The Nepal Earthquake Damage Assessment survey covers four districts in Nepal:
- Sindhupalchok
- Okhaldhunga
- Kavrepalanchok
- **Gorkha** (epicenter; focus of this project)

| Property | Value |
|---|---|
| Buildings surveyed | 234,835 |
| Households | 249,932 |
| Gorkha buildings (complete records) | 70,836 |
| Severe damage rate (Gorkha) | **64.26%** |
| Damage grades | 1–5 (Grade 4–5 = severe) |

Files:
- `nepal.sqlite` — full relational database (git-ignored, 125.7 MB)
- `data/building_structure.csv` — physical features of each building
- `data/building_damage.csv` — post-earthquake damage assessment
- `data/household_demographics.csv` — occupant information
- `data/id_map.csv` — bridge table linking buildings to households and districts

## Results (Lesson 4.1)

- Confirmed SQLite database contains exactly four tables: `building_structure`, `building_damage`, `household_demographics`, `id_map`.
- `building_structure` has 18 columns describing physical features. Five distinct foundation types: `Mud mortar-Stone/Brick` (91.5% of all buildings), `RC` (3.7%), `Cement-Stone/Brick` (3.4%), `Bamboo/Timber` (1.0%), `Other` (0.4%).
- Filtered to Gorkha (district_id = 4): **70,836 buildings** — exact match to the lesson's stated count. Zero orphaned foreign keys across all three joins.
- Binary target: `severe_damage` = 1 for Grade 4–5 damage, 0 for Grade 1–3.
- **Baseline accuracy is 64.26%** — any classifier in Lesson 4.2 must beat this. A naïve "always predict severe" classifier gets 64.26% accuracy without learning anything.

### A surprising benchmark result

The lesson material claims DuckDB is 10–100× faster than SQLite on analytical queries. **On our 234,835-row dataset, SQLite was 4.6× faster** (0.14s vs 0.63s) for a `GROUP BY foundation_type` aggregation query.

Two reasons:
- **CSV parse cost dominates on small data.** DuckDB reads raw CSV text and must parse it into typed columns before aggregating; SQLite reads a pre-built binary database.
- **Column storage advantage scales with data size.** DuckDB's memory-bandwidth savings become dominant at 10M+ rows; at 234K rows, everything fits in RAM.

DuckDB's convenience (querying CSVs directly, no database import step) still makes it the right choice for reproducibility — a reviewer only needs the CSVs, not a prebuilt `.sqlite` file. But the benchmark result is reported honestly here rather than glossed over.

## Results (Lesson 4.2)

- Trained a logistic regression model using a `Pipeline` with `OneHotEncoder` and `StandardScaler`.
- **Test Accuracy**: 70.09% (vs 64.40% baseline). Improvement: +5.68 percentage points.
- **Precision**: 69.10% | **Recall**: 95.92%.
- **Confusion Matrix**: 1,275 TN, 3,870 FP, 368 FN, 8,655 TP.
- **AUC**: 0.633 (weak discriminative power; only 13.3 pp better than random).

### Key Findings & Limitations

- **Model is aggressive, not discriminative**: The model correctly identifies 96% of severely damaged buildings (high recall), but it does so by flagging almost everything as "severe." The 3,870 false positives mean it struggles to identify safe buildings.
- **Low AUC (0.633)**: The model is barely better than random guessing at ranking buildings by risk. It learned that "Mud mortar = bad" and "RC = good", but cannot distinguish *which* Mud buildings will survive.
- **Class Imbalance**: Because 89.4% of buildings have Mud mortar foundations, the model easily learns the majority pattern but fails on the minority.
- **Odds Ratios confirm physical intuition**: RC foundation halves the odds of severe damage (OR = 0.59). Mud mortar increases them (OR = 1.38). However, features like age, height, and plinth area contribute almost nothing (OR ≈ 1.0).
- **Convergence fix**: Initial training failed to converge (1000 iterations) due to unscaled numeric features. Adding `StandardScaler` resolved this (converged in 10 iterations).


## Results (Lesson 4.3)

- Built a Decision Tree classifier with a three-way train/validation/test split (60/20/20).
- Hyperparameter tuning (testing max_depth 1-15) found **best depth = 5** (Validation Accuracy: 71.43%).
- **Test Accuracy**: **70.41%** (vs 64.41% baseline, +5.99 pp).
- **Comparison**: Decision Tree (70.41%) and Logistic Regression (70.09%) are statistically tied.

### Key Findings

- **Feature Importance**: `ground_floor_type` dominates with **67%** of the Gini importance. `foundation_type` is second at 16%. The root split isolates a small subset of "safe" buildings (likely RC-engineered ground floors), which drives most of the predictive power.
- **Bias-Variance Tradeoff**: The validation curve peaked at depth 5. At depth 15, training accuracy reached 75.5% but validation accuracy dropped to 70.4%, demonstrating classic overfitting.
- **Ordinal Encoding Caveat**: The tree uses `OrdinalEncoder`, which arbitrarily orders nominal categories like `foundation_type`. While the tree still learns effective splits, the specific hierarchy (e.g., RC = 0 vs Mud = 2) is an artifact of alphabetical encoding, not a physical ordering.


## Reproduce

    uv sync
    # Place the Nepal data files:
    #   - nepal.sqlite (project root)
    #   - data/building_structure.csv
    #   - data/building_damage.csv
    #   - data/household_demographics.csv
    #   - data/id_map.csv

    # Open notebooks/4.1_databases_sqlite_duckdb.ipynb

Tested with Python 3.14.2, pandas 3.1.0, numpy 2.5.4, duckdb 2.0.0.

## Notebooks

| Notebook | Topic |
|---|---|
| `notebooks/4.1_databases_sqlite_duckdb.ipynb` | SQLite vs DuckDB, relational joins, binary target creation |
| `notebooks/4.2_logistic_regression.ipynb` | EDA, OneHotEncoding, Logistic Regression, ROC/AUC evaluation |
| `notebooks/4.3_decision_trees.ipynb` | Decision Trees, validation curve, bias-variance tradeoff, feature importance |

## Layout

    data/                    raw CSV files (git-ignored)
    notebooks/               one notebook per lesson
    src/duckdb_wrangle.py    wrangle_nepal_data(): pipeline for Lesson 4.2+
    reports/figures/         saved charts
    pyproject.toml, uv.lock  environment (uv)
