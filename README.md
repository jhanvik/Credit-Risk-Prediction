# Credit Risk Prediction  

## Overview  
This project focuses on building a **credit risk prediction model** using machine learning techniques. The goal is to classify loan applicants as **low-risk or high-risk** based on financial and demographic attributes. This helps financial institutions in making informed lending decisions and reducing loan default risks.  

## Dataset  
The dataset includes various attributes related to loan applicants, such as:  
- **Applicant Income** – Monthly income of the applicant  
- **Credit History** – Record of past credit repayment behavior  
- **Loan Amount** – The requested loan amount  
- **Loan Term** – Duration of the loan  
- **Debt-to-Income Ratio** – Ratio of debt obligations to income  
- **Employment Status** – Employment type and years of experience  
- **Other Financial Indicators** – Such as savings, number of dependents, and past defaults  

The target variable categorizes applicants into:  
- **Low-Risk (1)** – Applicants with a high probability of repaying the loan  
- **High-Risk (0)** – Applicants with a higher chance of defaulting  

## Objectives  
- Perform **Exploratory Data Analysis (EDA)** to understand the distribution of credit risk factors.  
- Preprocess the dataset by handling missing values, encoding categorical features, and normalizing numerical attributes.  
- Train classification models to predict credit risk.  
- Evaluate model performance using key metrics such as **accuracy, precision, recall, and F1-score**.  

## Methodology  

1. **Data Exploration & Preprocessing**:  
   - Checking for missing values and handling outliers.  
   - Encoding categorical variables such as employment type and credit history.  
   - Standardizing numerical variables for optimal model performance.  

2. **Feature Engineering**:  
   - Selecting key financial indicators to improve predictive accuracy.  
   - Creating new features such as **loan-to-income ratio** and **credit utilization rate**.  

3. **Model Training**:  
   - Implemented multiple machine learning models, including:  
     - **Logistic Regression**  
     - **Decision Tree**  
     - **Random Forest**  
     - **Gradient Boosting (XGBoost, LightGBM)**  

4. **Model Evaluation**:  
   - Used **confusion matrix, ROC-AUC curve, and classification report** to compare model performance.  

## Outputs  

### 1. Data Insights  
- **Loan Amount vs. Credit Risk:** Showed higher default risk with increasing loan amounts.  
- **Income Distribution:** High-risk applicants had lower income levels on average.  

### 2. Model Performance  
| Model                 | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 82.5%    | 78.3%     | 80.2%  | 79.2%    |
| Decision Tree        | 86.1%    | 82.5%     | 83.7%  | 83.1%    |
| Random Forest       | **91.3%**    | **88.6%**  | **89.4%**  | **89.0%**    |
| Gradient Boosting   | 90.7%    | 87.9%     | 88.5%  | 88.2%    |

- **Random Forest** achieved the highest accuracy of **91.3%**, outperforming other models.  
- **Feature Importance Analysis** revealed that **credit history, debt-to-income ratio, and loan amount** were the most significant predictors.  
- **ROC-AUC Score:** The best model (Random Forest) had an **AUC of 0.94**, indicating strong predictive capability.  

### 3. Key Insights  
- Applicants with **good credit history** and **low debt-to-income ratios** were more likely to be classified as **low risk**.  
- Higher loan amounts and shorter employment durations correlated with **higher default risks**.  
- The **Random Forest model** provided the best trade-off between accuracy and generalization.  

## Technologies Used  
- Python  
- Pandas, NumPy, Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost, LightGBM  
- Jupyter Notebook  

## Conclusion  
This project successfully demonstrates how **machine learning can be used to assess credit risk**. By leveraging financial and demographic attributes, we can classify applicants effectively and reduce the likelihood of loan defaults.  

## Future Enhancements  
- Implement a **deep learning-based credit risk model** using neural networks.  
- Deploy the model as an API for real-time credit risk assessment.  
- Integrate additional external data sources (e.g., credit bureau scores) to improve accuracy.  
