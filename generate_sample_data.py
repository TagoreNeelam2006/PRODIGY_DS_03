"""
generate_sample_data.py
------------------------
Generates a synthetic dataset matching the structure and statistical
patterns of the UCI Bank Marketing dataset (bank-full.csv / bank.csv).

IMPORTANT: Replace bank_marketing.csv with the real dataset from:
https://archive.ics.uci.edu/dataset/222/bank+marketing

Official columns:
age, job, marital, education, default, balance, housing, loan, contact,
day, month, duration, campaign, pdays, previous, poutcome, y

Target column `y`: whether the client subscribed to a term deposit (yes/no).
decision_tree_classifier.py works identically against the real file since
column names match exactly.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 4000

jobs = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"]
job_probs = [0.11, 0.22, 0.03, 0.03, 0.21, 0.05, 0.03, 0.09, 0.02, 0.17, 0.03, 0.01]

marital = np.random.choice(["married", "single", "divorced"], size=n, p=[0.60, 0.28, 0.12])
job = np.random.choice(jobs, size=n, p=job_probs)
education = np.random.choice(["primary", "secondary", "tertiary", "unknown"],
                              size=n, p=[0.15, 0.51, 0.29, 0.05])
default = np.random.choice(["yes", "no"], size=n, p=[0.02, 0.98])
housing = np.random.choice(["yes", "no"], size=n, p=[0.56, 0.44])
loan = np.random.choice(["yes", "no"], size=n, p=[0.16, 0.84])
contact = np.random.choice(["cellular", "telephone", "unknown"], size=n, p=[0.65, 0.06, 0.29])
month = np.random.choice(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
                           "sep", "oct", "nov", "dec"], size=n,
                          p=[0.03, 0.04, 0.02, 0.06, 0.30, 0.11, 0.15, 0.11,
                             0.02, 0.03, 0.04, 0.09])
poutcome = np.random.choice(["success", "failure", "other", "unknown"],
                             size=n, p=[0.05, 0.11, 0.04, 0.80])

age = np.clip(np.random.normal(41, 11, n), 18, 95).astype(int)
balance = np.clip(np.random.normal(1350, 3000, n), -3000, 100000).astype(int)
day = np.random.randint(1, 29, n)
duration = np.clip(np.random.exponential(scale=260, size=n), 0, 4000).astype(int)
campaign = np.clip(np.random.poisson(2.5, n), 1, 40)
pdays = np.where(np.random.rand(n) < 0.82, -1, np.random.randint(1, 400, n))
previous = np.where(pdays == -1, 0, np.random.poisson(1.5, n))

# Target depends on several factors, mirroring real relationships:
# longer call duration, prior success, and being a student/retired
# correlate with higher subscription rates in the real dataset.
logit = (
    -2.4
    + 0.0035 * duration
    + np.where(poutcome == "success", 2.8, 0)
    + np.where(poutcome == "failure", -0.5, 0)
    + np.where(job == "student", 1.4, 0)
    + np.where(job == "retired", 1.1, 0)
    + np.where(job == "blue-collar", -0.6, 0)
    + np.where(housing == "no", 0.7, 0)
    + np.where(loan == "no", 0.4, 0)
    + np.where(contact == "cellular", 0.5, 0)
    + np.where(age < 25, 0.9, 0)
    + np.where(age > 60, 1.0, 0)
    + 0.00003 * balance
    - 0.12 * campaign
    + np.where(previous > 0, 0.6, 0)
)
prob = 1 / (1 + np.exp(-logit))
y = np.where(np.random.rand(n) < prob, "yes", "no")

df = pd.DataFrame({
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
    "y": y,
})

df.to_csv("bank_marketing.csv", index=False, sep=";")
print("Saved bank_marketing.csv:", df.shape)
print("\nTarget distribution:")
print(df["y"].value_counts(normalize=True).mul(100).round(1))
