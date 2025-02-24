# Credit Risk Prediction  

## Overview  
This project focuses on **predicting credit risk** using machine learning models to classify loan applicants as low or high risk. We implemented and compared multiple models, including **Random Forest, Support Vector Machine (SVM), and Logistic Regression**, to determine the most accurate approach.  

## Dataset  
We used a dataset containing historical credit data, including features such as income, loan amount, credit history, and other financial indicators. The dataset was preprocessed through **feature selection, normalization, and handling of missing values** before training the models.  

## Models and Performance  

### **Random Forest Classifier**  
- **Accuracy on Test Data:** **80.71%**  
- **Cross-Validation Score:** **78.57%**  

### **Support Vector Machine (SVM)**  
- **Initial Model Accuracy:** **79.29%**  
- **Optimized Model Accuracy (Best Parameters):** **82.14%**  
- **Best Parameters:** `C = 0.1, gamma = 0.1, kernel = 'linear'`  

### **Logistic Regression**  
- **Accuracy on Test Data:** **83.57%**  

## Evaluation  
- **Confusion Matrix & Heatmap Analysis:** Visualized misclassifications and overall model performance.  
- **Hyperparameter Tuning:** Used **GridSearchCV** for SVM to find the best-performing hyperparameters.  

## Conclusion  
- **Logistic Regression** achieved the highest accuracy (**83.57%**), making it the best model for this dataset.  
- **SVM with optimized parameters** improved performance significantly, reaching **82.14% accuracy**.  
- **Random Forest** provided a **robust baseline** but underperformed compared to the other models.  

## Future Improvements  
- Exploring **ensemble techniques** to combine model strengths.  
- Incorporating **feature engineering** to improve model interpretability.  
- Applying **deep learning methods** for further enhancement.  
