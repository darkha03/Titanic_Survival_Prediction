# 🏆 Kaggle Challenge Projects

A collection of machine learning projects solving real-world predictive challenges from Kaggle competitions. This repository showcases advanced data science techniques including exploratory analysis, feature engineering, and model optimization.

## 📂 Projects

### 1. 🏡 Ames House Prices: Advanced Regression Techniques
**Challenge:** Predict the final sale price of homes in Ames, Iowa  
**Model Performance:** MAE ~$15,000 (< 8.5% error on $180k homes)  
**Best Algorithm:** XGBoost Regressor

#### Key Highlights:
- **80+ features** analyzed using correlation heatmaps to identify price drivers
- **Data cleaning:** Handled categorical vs. missing data distinctions (e.g., "None" vs. NaN)
- **Feature engineering:** One-hot encoded 81 columns into 200+ features
- **Model comparison:** Random Forest, Linear Regression, and XGBoost tested
- **Insight:** Minor features matter—over-pruning weak columns actually increased error

**Notebooks:** [house_price.ipynb](house_price.ipynb) | [house_price.md](house_price.md)

---

### 2. 🚢 Titanic Survival Prediction
**Challenge:** Predict passenger survival based on demographics and ticket information  
**Model Performance:** 84.5% validation accuracy | 77% Kaggle test accuracy  
**Best Algorithm:** Random Forest Classifier (with hyperparameter tuning)

#### Key Highlights:
- **Strong predictors identified:** Gender and passenger class were dominant survival factors
- **Feature engineering:** Extracted titles (Mr, Mrs, Master) from names to capture historical context
- **Handling missing data:** Median imputation for age, dropped unreliable columns
- **Hyperparameter tuning:** GridSearchCV optimized n_estimators, max_depth, and min_samples_split
- **Model comparison:** Logistic Regression (70%), XGBoost (83%), Random Forest (84.5%)

**Notebooks:** [titanic.ipynb](titanic.ipynb) | [titanic.md](titanic.md)

---

## 🛠️ Tech Stack
- **Language:** Python 3
- **Data Processing:** Pandas
- **Machine Learning:** Scikit-Learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebooks


## 📊 Skills Demonstrated

- **Exploratory Data Analysis (EDA):** Correlation analysis, outlier detection, visualization
- **Data Cleaning:** Handling missing values, distinguishing between NaN and categorical "None"
- **Feature Engineering:** One-hot encoding, feature extraction, family size aggregation
- **Model Selection:** Testing multiple algorithms and selecting based on performance
- **Hyperparameter Tuning:** GridSearchCV for optimal model parameters
- **Error Analysis:** Understanding validation vs. test accuracy gaps
- **Regression & Classification:** Both supervised learning paradigms covered

## 📈 Results Summary

| Project | Task | Best Model | Score |
|---------|------|-----------|-------|
| House Prices | Regression | XGBoost | MAE: $15,000 |
| Titanic | Classification | Random Forest | 77% Accuracy |

## 📝 Notes

- Each project includes detailed markdown documentation explaining the methodology
- Notebooks are designed for educational purposes with step-by-step explanations
- Model performance reflects real Kaggle competition submissions

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas scikit-learn xgboost matplotlib seaborn notebook
   ```

3. **Download datasets from Kaggle:**
   - [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
   - [Titanic](https://www.kaggle.com/c/titanic)

4. **Run the notebooks:**
   ```bash
   jupyter notebook
   ```