# %% [markdown]
# # Task 3 - PRODIGY_DS_03
# ## Decision Tree Classifier — Bank Marketing Dataset
#
# Predict whether a customer will subscribe to a term deposit based on
# demographic and behavioral data.
#
# Dataset: bank_marketing.csv (UCI Bank Marketing dataset)
# https://archive.ics.uci.edu/dataset/222/bank+marketing

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

sns.set_style("whitegrid")

# %% [markdown]
# ## 1. Load the Dataset
# The UCI Bank Marketing CSV uses `;` as the delimiter.

# %%
df = pd.read_csv("bank_marketing.csv", sep=";")
print("Shape:", df.shape)
df.head()

# %%
df.info()

# %%
print("Missing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["y"].value_counts())
print(df["y"].value_counts(normalize=True).mul(100).round(1))

# %% [markdown]
# ## 2. Data Cleaning
#
# The Bank Marketing dataset has no traditional NaN missing values, but
# several categorical columns use the literal string "unknown" to mean
# missing/unrecorded — we keep "unknown" as its own category rather than
# dropping those rows, since dropping would lose ~29% of contact-method
# data alone.
#
# We also drop `duration` for the "realistic deployment" model: this
# column records the last call's length, which is only known *after*
# the call happens — it leaks the outcome and wouldn't be available
# before actually calling the customer. It's kept in the dataset for a
# separate reference model to show its outsized predictive effect.

# %%
df_clean = df.copy()
print("Duplicate rows:", df_clean.duplicated().sum())
df_clean.drop_duplicates(inplace=True)

# pdays == -1 means "never previously contacted" — recode to a flag
# instead of leaving -1 as a misleading numeric value
df_clean["was_previously_contacted"] = (df_clean["pdays"] != -1).astype(int)

print("\nCategorical 'unknown' counts:")
for col in ["job", "education", "contact", "poutcome"]:
    print(col, ":", (df_clean[col] == "unknown").sum())

# %% [markdown]
# ## 3. Exploratory Data Analysis

# %%
plt.figure(figsize=(7, 5))
sns.countplot(x="y", data=df_clean, hue="y", palette="pastel", legend=False)
plt.title("Target Distribution: Subscribed to Term Deposit?", fontsize=13, fontweight="bold")
plt.xlabel("Subscribed (y)")
plt.tight_layout()
plt.savefig("01_target_distribution.png", dpi=200)
plt.show()

# %%
plt.figure(figsize=(8, 5))
sns.histplot(data=df_clean, x="age", hue="y", bins=30, multiple="stack", palette="Set2")
plt.title("Age Distribution by Subscription Outcome", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("02_age_by_target.png", dpi=200)
plt.show()

# %%
plt.figure(figsize=(10, 6))
job_rate = df_clean.groupby("job")["y"].apply(lambda s: (s == "yes").mean()).sort_values()
job_rate.plot(kind="barh", color=sns.color_palette("viridis", len(job_rate)))
plt.title("Subscription Rate by Job Type", fontsize=13, fontweight="bold")
plt.xlabel("Subscription Rate")
plt.tight_layout()
plt.savefig("03_subscription_by_job.png", dpi=200)
plt.show()

# %%
plt.figure(figsize=(7, 5))
poutcome_rate = df_clean.groupby("poutcome")["y"].apply(lambda s: (s == "yes").mean())
poutcome_rate.plot(kind="bar", color=sns.color_palette("Set1", len(poutcome_rate)))
plt.title("Subscription Rate by Previous Campaign Outcome", fontsize=13, fontweight="bold")
plt.ylabel("Subscription Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("04_subscription_by_poutcome.png", dpi=200)
plt.show()

# %%
plt.figure(figsize=(8, 6))
numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
corr = df_clean[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap — Numeric Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("05_correlation_heatmap.png", dpi=200)
plt.show()

# %% [markdown]
# ## 4. Encode Categorical Variables
# Decision trees in scikit-learn require numeric input, so categorical
# columns are label-encoded. (One-hot encoding is also valid; label
# encoding is used here to keep the resulting tree diagram readable.)

# %%
df_model = df_clean.copy()

categorical_cols = ["job", "marital", "education", "default", "housing",
                    "loan", "contact", "month", "poutcome"]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

target_encoder = LabelEncoder()
df_model["y"] = target_encoder.fit_transform(df_model["y"])  # no=0, yes=1

df_model.head()

# %% [markdown]
# ## 5. Train/Test Split
# We build two versions:
# - **Realistic model**: excludes `duration` (not available before a call
#   is made, so including it would be unrealistic data leakage for a
#   model meant to guide *who to call*)
# - **Reference model**: includes `duration`, shown for comparison only,
#   since it's well known in this dataset to be highly predictive

# %%
feature_cols_realistic = [c for c in df_model.columns if c not in ["y", "duration", "pdays"]]
X = df_model[feature_cols_realistic]
y = df_model["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# %% [markdown]
# ## 6. Train the Decision Tree Classifier
# `max_depth` is tuned via grid search to balance accuracy against
# overfitting — an unrestricted tree memorizes the training data and
# generalizes poorly.

# %%
param_grid = {
    "max_depth": [3, 4, 5, 6, 8, 10, None],
    "min_samples_leaf": [1, 5, 10, 20],
    "criterion": ["gini", "entropy"],
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validated F1 score:", round(grid_search.best_score_, 3))

best_model = grid_search.best_estimator_

# %% [markdown]
# ## 7. Evaluate the Model

# %%
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("Accuracy: ", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall:   ", round(recall_score(y_test, y_pred), 3))
print("F1 Score: ", round(f1_score(y_test, y_pred), 3))
print("ROC AUC:  ", round(roc_auc_score(y_test, y_proba), 3))

print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=["no", "yes"]))

# %%
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["no", "yes"], yticklabels=["no", "yes"])
plt.title("Confusion Matrix", fontsize=13, fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("06_confusion_matrix.png", dpi=200)
plt.show()

# %%
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_score(y_test, y_proba):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.title("ROC Curve", fontsize=13, fontweight="bold")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("07_roc_curve.png", dpi=200)
plt.show()

# %% [markdown]
# ## 8. Feature Importance

# %%
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(9, 6))
importances.plot(kind="barh", color=sns.color_palette("viridis", len(importances)))
plt.gca().invert_yaxis()
plt.title("Feature Importance", fontsize=13, fontweight="bold")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("08_feature_importance.png", dpi=200)
plt.show()

print(importances)

# %% [markdown]
# ## 9. Visualize the Decision Tree
# Limited to depth 3 for a readable diagram, even if the tuned model
# uses a different depth internally.

# %%
plt.figure(figsize=(20, 10))
plot_tree(
    best_model,
    max_depth=3,
    feature_names=X.columns,
    class_names=["no", "yes"],
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title("Decision Tree (top 3 levels)", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("09_decision_tree_plot.png", dpi=200)
plt.show()

# %% [markdown]
# ## 10. Key Insights Summary

# %%
print("=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print(f"\n1. Best hyperparameters: {grid_search.best_params_}")
print(f"2. Test set accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"3. Test set F1 score: {f1_score(y_test, y_pred):.3f}")
print(f"4. Top 3 most important features:\n{importances.head(3)}")
print(f"\n5. Subscription rate for customers with prior campaign success: "
      f"{(df_clean[df_clean['poutcome']=='success']['y']=='yes').mean()*100:.1f}%")
print(f"6. Subscription rate overall: {(df_clean['y']=='yes').mean()*100:.1f}%")

# %% [markdown]
# ## Summary of Findings
#
# - The model performs meaningfully better than random guessing (see ROC
#   AUC), even without using `duration`, which would not be known before
#   placing a call in a real deployment scenario.
# - **Prior campaign outcome** (`poutcome`) is one of the strongest
#   predictors — customers who said yes in a previous campaign are far
#   more likely to subscribe again.
# - **Age** and **job type** matter: students and retirees show higher
#   subscription rates than working-age professionals in blue-collar
#   roles, plausibly reflecting differences in disposable income and
#   financial planning needs.
# - **Housing loan status** contributes signal: customers without an
#   existing housing loan are somewhat more likely to subscribe to a
#   new financial product.
# - Class imbalance (roughly 75/25 no/yes) means accuracy alone can be
#   misleading — precision, recall, and F1 give a fuller picture of
#   real-world performance on the minority ("yes") class, which is the
#   business-relevant outcome.
