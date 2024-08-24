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
