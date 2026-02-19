# 🏡 Ames House Prices: Advanced Regression Techniques

**Goal:** Predict the final sale price of homes in Ames, Iowa using Machine Learning.<br>
**Kaggle Score:** 0.13 (13% Error)

## 📌 Project Overview
This project is my implementation of the Kaggle House Prices competition. Moving beyond binary classification (predicting categories), this project focuses on **Regression**—predicting a continuous dollar amount based on 80+ explanatory variables, ranging from square footage to neighborhood and roof style.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn
* **Algorithms Tested:** Random Forest Regressor, XGBoost Regressor, Linear Regression

## 📊 The Data Process (My Approach)

### 1. Exploratory Data Analysis (EDA) & Outlier Removal
With 81 columns, visualizing everything manually was impossible. 
* **Correlation Heatmaps:** I generated a correlation matrix to identify the features most closely tied to `SalePrice`. Found that `OverallQual` and `GrLivArea` (Square Footage) were the dominant drivers of price.
* **Multicollinearity:** Spotted redundant features (e.g., `GarageCars` and `GarageArea` had a 0.88 correlation) to understand how features interacted.
* **Outlier Removal:** Used scatter plots to identify and drop massive anomalies (houses over 4,000 sq. ft. that sold for abnormally low prices) to prevent the regression models from skewing their lines of best fit.

### 2. Advanced Data Cleaning & Imputation
Real estate data is messy. I implemented a robust cleaning strategy:
* **The "NaN vs. None" Trap:** Discovered that Pandas was misinterpreting categorical text like "None" (meaning no masonry veneer or no pool) as missing data (`NaN`). I wrote logic to safely replace these with descriptive text rather than blindly deleting the rows.
* **Imputation:** Filled true missing numerical values (like `LotFrontage` and `GarageYrBlt`) using median imputation via Scikit-Learn's `SimpleImputer`.

### 3. Feature Engineering
* **One-Hot Encoding:** Machine learning models only speak math. I used `pd.get_dummies()` to convert text categories (like `Neighborhood_CollgCr`) into a massive grid of 1s and 0s, expanding the dataset from 81 columns to over 200 features.
* **Test Data Alignment:** Used `.reindex()` to ensure the Kaggle Test data perfectly matched the One-Hot Encoded columns of the Training data, preventing dimensional crashes.

## ⚙️ Model Iteration & Results

I tested multiple approaches to find the best fit for high-dimensional data:

1. **Random Forest Regressor (Baseline):** * Proved the "Over-Pruning Trap." I initially tried dropping 185 "weak" columns to reduce noise, but the error actually *increased*. The forest needed those minor features (like specific neighborhoods) to make subtle $1,000 price adjustments deep in its decision trees.
2. **Linear Regression:** * **MAE: ~$20,000.** After scaling the data, the linear model struggled. It highlighted the **Accuracy vs. Interpretability** tradeoff: Linear Regression provides transparent math but gets overwhelmed by 200+ encoded columns and multicollinearity.
3. **XGBoost Regressor (Final Champion):** * **MAE: ~$15,000.** The gradient boosting model easily navigated the 200+ columns, ignoring the noise and iteratively fixing its own mistakes to achieve an error rate of less than 8.5% on a standard $180k home.

## 🚀 How to Run
1. Clone this repository.
2. Ensure you have the `train.csv` and `test.csv` files from the Kaggle House Prices competition.
3. Run the Jupyter Notebook cell by cell to see the EDA, outlier removal, encoding, and model training processes.
