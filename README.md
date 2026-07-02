# Student Performance Predictive Modeling

A **full end-to-end machine learning pipeline** predicting student academic outcomes from demographic and behavioral features, producing portfolio-ready metrics and visualizations.

---

## 📊 Key Results

| Model | R² | RMSE | MAE | CV R² | Error Reduction |
|---|---|---|---|---|---|
| **Ridge Regression** ⭐ | **0.6917** | **5.33** | **4.21** | 0.6935 ± 0.027 | **44.5%** |
| Random Forest | 0.6671 | 5.53 | 4.36 | 0.6449 ± 0.042 | 42.3% |
| Gradient Boosting | 0.6492 | 5.68 | 4.56 | 0.6103 ± 0.039 | 40.8% |
| Baseline (mean) | -0.0002 | 9.59 | 7.67 | — | — |

> Ridge Regression achieved the best generalisation: **R² = 0.69**, reducing prediction error by **44.5%** vs. naive baseline.

---

## 🚀 Resume Bullet

> *"Developed and validated a Ridge Regression model predicting student academic outcomes (R² = 0.69, RMSE = 5.33); reduced prediction error by 44% vs. baseline mean predictor; implemented full end-to-end ML pipeline — feature engineering, 5-fold cross-validation, and rigorous model comparison across 3 algorithms."*

---

## 📂 Project Structure

```
Student-Performance-Predictive-Modeling/
│
├── data/
│   ├── generate_dataset.py     # Generates realistic 1 000-row dataset
│   └── StudentsPerformance.csv # Generated dataset
│
├── eda.py                      # Exploratory Data Analysis (7 visuals)
├── model.py                    # Model training + evaluation (6 visuals)
├── metrics.txt                 # Saved quantitative metrics report
│
└── visuals/
    ├── eda/                    # EDA visualizations
    │   ├── 01_score_distributions.png
    │   ├── 02_correlation_heatmap.png
    │   ├── 03_avg_score_by_category.png
    │   ├── 04_study_absences_scatter.png
    │   ├── 05_parental_education_score.png
    │   ├── 06_boxplot_prep_gender.png
    │   └── 07_absences_log_transform.png
    └── model/                  # Model visualizations
        ├── A_model_comparison.png
        ├── B_feature_importance.png
        ├── C_predicted_vs_actual.png
        ├── D_residuals_analysis.png
        ├── E_cross_validation.png
        └── F_all_models_pva.png
```

---

## 🛠️ Setup & Usage

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn scipy

# 2. Generate dataset
python data/generate_dataset.py

# 3. Run EDA
python eda.py

# 4. Train models & evaluate
python model.py
```

---

## 📈 Methodology

### Data & Features
- **Dataset**: 1,000 student records, 10 features, 0 missing values
- **Target**: `avg_score` = mean(math, reading, writing scores)
- **Features used**:
  - Ordinal: `parental_level_of_education` (6 levels)
  - Nominal (OHE): `gender`, `race_ethnicity`, `lunch`, `test_preparation_course`
  - Numerical: `weekly_study_hours`, `log1p(absences)`

### Feature Engineering
- **Composite target**: average of 3 subject scores
- **Log transformation**: absences normalized with `log1p` (reduces skewness from 1.14 → 0.38)
- **Ordinal encoding**: parental education encoded by level (0–5)
- **Standard scaling**: all numerical features z-score normalized

### Model Validation
- **80/20 train/test split** (stratified by score decile)
- **5-fold cross-validation** on training set for unbiased estimates
- Baseline: mean predictor (RMSE = 9.59)

### Key Insights from EDA
- Study hours correlate most strongly with performance (**r = 0.61**)
- Test prep course completion boosts average score by ~8 points
- Standard lunch students score ~7 points higher on average
- Absences weakly but negatively impact scores (r = −0.19)
- Parental education shows moderate positive effect (r = 0.20)

---

## 🔬 EDA Visualizations

| Plot | Insight |
|---|---|
| Score Distributions | Math/reading/writing scores are roughly normal (skew < 0.3) |
| Correlation Heatmap | Subject scores highly intercorrelated (r ≈ 0.85) |
| Category Bar Charts | Test prep & lunch type show clear score gaps |
| Study vs Score Scatter | Linear trend with r = 0.61 |
| Parental Education | Master's degree parents → ~8 pt higher avg |
| Boxplots (prep × gender) | Female students score higher in reading/writing |
| Absences Transform | log1p reduces skewness from 1.14 to 0.38 |

---

## Technologies

`Python 3.12` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn` · `SciPy`