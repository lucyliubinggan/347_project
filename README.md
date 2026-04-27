# NYC Airbnb Price Prediction
**DATASCI 347 · Spring 2026**

End-to-end supervised regression pipeline predicting NYC Airbnb nightly prices using the [2019 Kaggle dataset](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) (48,895 listings). Target variable is `log_price` (natural log of USD nightly price).

---

## Results

| Model | Test R² | MAE ($) |
|---|---|---|
| Linear Regression | 0.5954 | 56.84 |
| Ridge (α=48.33) | 0.5954 | 56.84 |
| Lasso (α=0.00044) | 0.5956 | 56.83 |
| Random Forest | 0.5879 | 56.96 |
| **XGBoost** | **0.6559** | **52.91** |

Top predictors across all models: **room type**, **neighbourhood**, and **longitude**.

---

## Project Structure

```
├── data/
│   ├── AB_NYC_2019.csv                  # raw data (48,895 listings)
│   ├── AB_NYC_2019_clean.csv            # after EDA cleaning
│   ├── AB_NYC_2019_text_features.csv    # after text feature extraction
│   ├── test_predictions.csv             # model predictions on test set
│   ├── model_comparison.csv             # R², RMSE, MAE for all models
│   ├── linear_coefficients.csv          # standardised coefficients
│   └── rf_feature_importance.csv        # Random Forest feature importance
├── figures/                             # all output plots
├── NYC_Airbnb_EDA.ipynb                 # Stage 1: cleaning & EDA
├── NYC_Airbnb_Text_Features.ipynb       # Stage 2: text feature engineering
├── NYC_Airbnb_Modelling.ipynb           # Stage 3: modelling & evaluation
└── NYC_Airbnb_Presentation.pptx
```

---

## Pipeline

Run the three notebooks **in order**:

### 1. `NYC_Airbnb_EDA.ipynb`
- Removes listings with `price = 0` or `minimum_nights > 365` (25 rows dropped)
- Fills `reviews_per_month` with 0 where `number_of_reviews == 0` (MNAR)
- Log-transforms price, reducing skewness from 19.1 → 0.6
- Produces `data/AB_NYC_2019_clean.csv`

### 2. `NYC_Airbnb_Text_Features.ipynb`
- Extracts 12 binary keyword flags from listing names (luxury, cozy, private, etc.)
- Extracts `bedroom_count` via regex, imputed with median (1.0)
- Applies TF-IDF (200 features, bigrams) then PCA to 128 components (80% variance)
- Produces `data/AB_NYC_2019_text_features.csv` (48,870 × 142 features)

### 3. `NYC_Airbnb_Modelling.ipynb`
- One-hot encodes `room_type` and `neighbourhood_group`
- Target-encodes `neighbourhood` (221 levels, smoothing λ=20)
- Standardises features for linear models; passes raw features to tree models
- Trains OLS, Ridge, Lasso (3-fold CV), Random Forest, and XGBoost (early stopping)
- Produces predictions, model comparison, and feature importance CSVs

---

## Dependencies

```
pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, wordcloud, xgboost
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn wordcloud xgboost
```

---

## Key Findings

- All linear models converge near R² ≈ 0.60 — regularisation brings no meaningful gain because n >> p and PCA already mitigated collinearity
- Random Forest scores slightly below linear models — the log-transform linearised the main signal, removing RF's main advantage
- XGBoost gains ~6% R² over linear models via sequential residual correction
- The remaining ~40% unexplained variance reflects unmeasured factors: photo quality, amenities, host responsiveness
