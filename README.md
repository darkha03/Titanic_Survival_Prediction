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

## ⚙️ Model Training & Results
I split the training data (80/20) to validate my model locally before submitting it to Kaggle.
* **Algorithm:** Random Forest Classifier 
* **Validation Accuracy:** ~84%
* **Final Kaggle Test Accuracy:** 77%
### Model Comparison
I tested both Random Forest and Logistic Regression (with Feature Scaling). The Random Forest outperformed Logistic Regression (84% vs 70%), indicating that the survival patterns in the Titanic dataset are highly non-linear and rely on complex conditional rules that a linear model cannot easily capture.
The slight drop between validation and test accuracy demonstrates mild overfitting, which is expected. The model successfully generalized the core patterns of the dataset.

## 🚀 How to Run
1. Clone this repository.
2. Ensure you have the `train.csv` and `test.csv` files from Kaggle.
3. Run the Jupyter Notebook cell by cell to see the EDA, cleaning, and model training process.
