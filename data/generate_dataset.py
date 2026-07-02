"""
Generate a realistic Student Performance dataset.
Mirrors the structure of the Kaggle Student Performance dataset
(UCI / StudentsPerformance.csv) with extended features.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1000

# ── Demographics ──────────────────────────────────────────────────────────────
gender = np.random.choice(["female", "male"], N, p=[0.52, 0.48])
race = np.random.choice(
    ["group A", "group B", "group C", "group D", "group E"],
    N, p=[0.09, 0.19, 0.32, 0.26, 0.14]
)
parental_education = np.random.choice(
    ["some high school", "high school", "some college",
     "associate's degree", "bachelor's degree", "master's degree"],
    N, p=[0.18, 0.32, 0.22, 0.12, 0.12, 0.04]
)
lunch = np.random.choice(["standard", "free/reduced"], N, p=[0.65, 0.35])
test_prep = np.random.choice(["none", "completed"], N, p=[0.64, 0.36])

# ── Study & attendance ────────────────────────────────────────────────────────
# weekly study hours: log-normal
study_hours = np.clip(np.random.lognormal(mean=1.5, sigma=0.6, size=N), 0, 20)

# absences: zero-inflated Poisson
absences = np.where(
    np.random.random(N) < 0.3,
    0,
    np.random.poisson(lam=4, size=N)
)
absences = np.clip(absences, 0, 30)

# ── Numeric boosts from categorical features ──────────────────────────────────
edu_map = {
    "some high school": 0, "high school": 1, "some college": 2,
    "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5
}
edu_score = np.array([edu_map[e] for e in parental_education])  # 0-5

gender_boost = np.where(gender == "female", 2.0, 0.0)
lunch_boost  = np.where(lunch == "standard", 5.0, 0.0)
prep_boost   = np.where(test_prep == "completed", 8.0, 0.0)
edu_boost    = edu_score * 1.5
study_boost  = study_hours * 2.0
absence_pen  = absences * 0.8

base_score = 55 + gender_boost + lunch_boost + prep_boost + edu_boost + study_boost - absence_pen

# ── Subject scores with individual noise ─────────────────────────────────────
math_score    = np.clip(base_score + np.random.normal(0, 10, N), 0, 100).astype(int)
reading_score = np.clip(base_score + np.random.normal(2, 9,  N), 0, 100).astype(int)
writing_score = np.clip(base_score + np.random.normal(1, 9,  N), 0, 100).astype(int)

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "gender":                    gender,
    "race_ethnicity":            race,
    "parental_level_of_education": parental_education,
    "lunch":                     lunch,
    "test_preparation_course":   test_prep,
    "weekly_study_hours":        np.round(study_hours, 1),
    "absences":                  absences,
    "math_score":                math_score,
    "reading_score":             reading_score,
    "writing_score":             writing_score,
})

os.makedirs("data", exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "StudentsPerformance.csv")
df.to_csv(out, index=False)
print(f"Dataset saved -> {out}  ({len(df)} rows, {df.shape[1]} columns)")
print(df.describe())
