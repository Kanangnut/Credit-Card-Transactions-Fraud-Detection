### Credit Card Fraud Detection using Machine Learning

This project focuses on building a robust fraud detection model using credit card transaction data. The goal is to predict fraudulent transactions based on transaction details and cardholder information. The project goes through several stages, from data preprocessing and exploratory data analysis (EDA) to modeling and evaluation using multiple machine learning algorithms. The result of this you can find on Pandas Profiling [Pandas Profiling](https://github.com/Kanangnut/Credit-Card-Transactions-Fraud-Detection/blob/main/pandas_profile_folder/Credit_Card_Transactions_Fraud_profile.html).
#### Contents

1. **Data Preprocessing**  
   - Imported necessary libraries like pandas, numpy, and various visualization tools.
   - Loaded the training and test data from CSV files.
   - Cleaned the data, addressing null values and identifying key features such as transaction time and amounts.

2. **Exploratory Data Analysis (EDA)**  
   - Conducted univariate and multivariate analyses to explore the relationships between features (e.g., transaction amount, gender, and category) and the target variable (`is_fraud`).
   - Created insightful visualizations such as category distributions, age variations, and fraud likelihood based on transaction amounts.

3. **Feature Engineering**  
   - Extracted additional features from the transaction timestamp such as time of day, month, and year.
   - Calculated the distance between the cardholder's home and the merchant's location.
   - Used one-hot encoding to transform categorical features into numerical form.

4. **Modeling**  
   - Trained and evaluated five machine learning models:
     - **KNN Model**: Achieved strong results on the training data but struggled to generalize on the test set.
     - **Gradient Boosting Model**: Delivered excellent results with a well-balanced performance across metrics.
     - **LightGBM Model**: Proved to be the best-performing model, achieving high accuracy and recall.
     - **Logistic Regression Model**: Performed decently but was outshined by more advanced models.
     - **Random Forest Model**: Showed very high accuracy but risked overfitting on the training data.

5. **Conclusion**  
   - After evaluating all models, the **LightGBM** and **Gradient Boosting** models stood out as the best performers, balancing accuracy and precision while managing the challenge of imbalanced data.

### Tools and Technical Stack

#### **Libraries/Frameworks Used:**

1. **Data Manipulation and Analysis:**
   - `pandas`: For data manipulation, cleaning, and exploratory data analysis.
   - `numpy`: For numerical computations and handling arrays.

2. **Data Visualization:**
   - `matplotlib`: For creating plots, histograms, and charts.
   - `seaborn`: For statistical data visualization, creating attractive plots.

3. **Missing Data Visualization:**
   - `missingno`: For visualizing missing data patterns.

4. **Feature Engineering and Encoding:**
   - `scikit-learn`: For feature scaling, train-test splitting, and machine learning models.
   - `category_encoders`: For categorical data encoding (e.g., WOE encoding).
   - `pandas-profiling`: For automated exploratory data analysis.

5. **Machine Learning Models:**
   - **K-Nearest Neighbors (KNN)**: For classification tasks.
   - **Random Forest**: For ensemble learning and feature importance.
   - **Gradient Boosting**: For boosting model performance.
   - **LightGBM**: For fast, efficient boosting models.
   - **Logistic Regression**: For baseline classification and binary prediction.

6. **Model Evaluation and Metrics:**
   - `scikit-learn`: For model evaluation metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.
   - `imblearn`: For handling imbalanced datasets, using undersampling techniques.

7. **Profiling and Feature Selection:**
   - `BorutaPy`: For feature importance and selection with random forests.

8. **Mathematical Operations and Statistical Functions:**
   - `scipy`: For statistical functions, probability distributions, and normality checks.
   - `math`: For additional mathematical operations.

9. **Date and Time Manipulation:**
   - `datetime`: For working with time-related features in the dataset.

#### **Data Science Workflow:**

- **Data Cleaning:** Removed duplicates, handled missing values, and performed feature engineering (e.g., time-based features, distance calculations).
- **Exploratory Data Analysis (EDA):** Visualized data distribution and relationships between key features and the target variable.
- **Feature Engineering:** Extracted relevant features (e.g., transaction time, age group, distance between home and merchant).
- **Model Building:** Implemented multiple machine learning models to predict fraudulent transactions.
- **Evaluation:** Assessed models using metrics like accuracy, recall, precision, F1-score, and ROC-AUC.
