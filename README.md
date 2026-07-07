# PRODIGY_DS_03

**Data Science Internship — Task 3**
Prodigy InfoTech

## 📌 Task
Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data, using the Bank Marketing dataset from the UCI Machine Learning Repository.

## 📊 Dataset
**Bank Marketing dataset** — [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing). Data from a Portuguese bank's direct marketing campaigns (phone calls) to predict whether a client subscribes to a term deposit.

Columns:
| Column | Description |
|---|---|
| age | Client age |
| job | Type of job |
| marital | Marital status |
| education | Education level |
| default | Has credit in default? |
| balance | Average yearly account balance |
| housing | Has housing loan? |
| loan | Has personal loan? |
| contact | Contact communication type |
| day, month | Last contact date |
| duration | Last contact duration (seconds) |
| campaign | # contacts during this campaign |
| pdays | Days since last contact from a previous campaign (-1 = never) |
| previous | # contacts before this campaign |
| poutcome | Outcome of previous campaign |
| **y** | **Target** — subscribed to a term deposit? (yes/no) |

> **Note:** `bank_marketing.csv` here is a synthetic dataset built to match the real UCI dataset's structure and general relationships (e.g. previous campaign success and student/retired status increasing subscription likelihood). Swap in the real file from the link above before final submission — no code changes needed, since column names and the `;` delimiter match exactly.

## 🛠️ Tech Stack
- Python 3
- pandas, numpy — data loading & cleaning
- scikit-learn — decision tree model, grid search, evaluation metrics
- matplotlib, seaborn — visualization

## 📁 Repository Structure
```
PRODIGY_DS_03/
├── generate_sample_data.py           # creates the sample bank_marketing.csv
├── bank_marketing.csv                 # dataset (swap with the real file)
├── decision_tree_classifier.py       # full cleaning + EDA + model + evaluation
├── 01_target_distribution.png … 09_decision_tree_plot.png   # generated charts
└── README.md
```

## ▶️ How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

# (optional) regenerate the sample dataset
python generate_sample_data.py

# run the full pipeline
python decision_tree_classifier.py
```
The script is structured with `# %%` cell markers, so it also runs cell-by-cell in VS Code (with the Jupyter extension) or in a Jupyter notebook.

## 🧹 Data Cleaning & Preparation
- Checked and removed exact duplicate rows
- `pdays` recoded: `-1` (never previously contacted) converted into a `was_previously_contacted` binary flag, since -1 is a placeholder, not a meaningful numeric distance
- Categorical `"unknown"` values kept as their own category rather than dropped, since dropping would discard a large share of rows (e.g. ~29% of `contact` in the sample)
- Categorical columns label-encoded for the model
- **`duration` deliberately excluded from the predictive model**: this column records the last call's length, which is only known *after* placing the call — including it would leak information not available at prediction time in a real "who should we call" use case

## 🌳 Modeling Approach
- `DecisionTreeClassifier` from scikit-learn, tuned via `GridSearchCV` (5-fold CV, optimizing F1 score) over `max_depth`, `min_samples_leaf`, and split criterion
- `class_weight="balanced"` used to address the class imbalance (~60/40 no/yes in the sample; real dataset is more imbalanced, closer to 88/12)
- Evaluated with accuracy, precision, recall, F1, ROC AUC, confusion matrix, and ROC curve — not accuracy alone, since accuracy is misleading under class imbalance
- Feature importance and a depth-limited tree diagram included for interpretability

## 📈 Results (on sample data — rerun against the real dataset for final numbers)
- Best parameters selected via grid search (see console output / top of `decision_tree_classifier.py` run)
- Test accuracy and F1 score printed at the end of the run, along with a full classification report

## 💡 Key Insights
- **Previous campaign outcome** is one of the strongest predictors — customers who subscribed in a past campaign are substantially more likely to subscribe again.
- **Job type and age** matter: students and retirees show higher subscription rates than working-age professionals in blue-collar roles.
- **Existing housing/personal loans** correlate with lower subscription likelihood, plausibly reflecting less available disposable income.
- Excluding `duration` (a leakage-prone feature) gives a more realistic picture of what's actually predictable *before* making a call, which is the practically useful version of this model for campaign targeting.

## 🔗 About
This repository is part of my Data Science Internship at **Prodigy InfoTech**.

#DataScience #Python #MachineLearning #DecisionTree #ProdigyInfoTech #Internship
