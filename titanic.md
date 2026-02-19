# 🚢 Titanic Survival Prediction

**Goal:** Predict passenger survival on the Titanic using Machine Learning.<br>
**Kaggle Score:** 0.77 (77% Accuracy)

## 📌 Project Overview
This project is my implementation of the classic Kaggle Titanic Machine Learning challenge. I built a binary classification model to predict whether a passenger survived or died based on features like age, gender, ticket class, and family size.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn
* **Model:** Random Forest Classifier and Logistic Regression

## 📊 The Data Process (My Approach)

### 1. Exploratory Data Analysis (EDA)
I started by visualizing the data to find the strongest predictors of survival. 
* Found that **Gender** was the strongest baseline predictor (females survived at a much higher rate).
* Found that **Passenger Class** heavily influenced survival (1st class had better odds than 3rd class).

### 2. Data Cleaning
Machine learning models require clean, numerical data.
* **Age:** Filled missing values (177 rows) using the median age to prevent skewed data.
* **Cabin:** Dropped this column entirely because ~77% of the data was missing, making it unreliable.
* **Embarked:** Dropped the 2 rows with missing departure ports.
* **Categorical Encoding:** Mapped text values (e.g., 'male'/'female') to binary integers (0/1).

### 3. Feature Engineering 💡
Instead of relying only on the raw data, I created new features to give the model better context:
* **Title Extraction:** Extracted titles (Mr, Mrs, Miss, Master) from the `Name` column. This allowed the model to differentiate between adult men (Mr) and male children (Master), capturing the "women and children first" historical context.
* **Family Size:** Combined `SibSp` (siblings/spouse) and `Parch` (parents/children) to see how family dynamics affected survival.

## ⚙️ Model Training, Tuning & Results
I split the training data (80/20) to validate my models locally before submitting to Kaggle. 

### Model Comparison
I tested three distinct algorithms to find the best fit for the data's structure:
1. **Logistic Regression (with Feature Scaling):** Achieved ~70% accuracy. The linear nature of this model struggled to capture the highly conditional rules of survival.
2. **XGBoost:** Achieved ~83% accuracy. While powerful, the small size of the dataset limited the boosting algorithm's ability to outperform simpler ensemble methods without overfitting.
3. **Random Forest Classifier:** Outperformed the others with an **~84.5% validation accuracy**. The decision tree ensemble perfectly captured the non-linear, categorical nature of the Titanic dataset.

### Hyperparameter Tuning

To push the Random Forest model to its absolute limit, I utilized `GridSearchCV` to automate the tuning of the model's hyperparameters. By systematically testing combinations of `n_estimators`, `max_depth`, and `min_samples_split`, I optimized the model to balance high accuracy with resistance to overfitting.

### Final Results
* **Validation Accuracy:** ~84.5%
* **Final Kaggle Test Accuracy:** 77%

The slight drop between validation and test accuracy demonstrates mild overfitting, which is expected. The model successfully generalized the core patterns of the dataset, but ultimately reached the threshold of "irreducible error" caused by the random nature of the historical event.

## 🚀 How to Run
1. Clone this repository.
2. Ensure you have the `train.csv` and `test.csv` files from Kaggle.
3. Run the Jupyter Notebook cell by cell to see the EDA, cleaning, and model training process.
