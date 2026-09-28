# Gym Customer Churn Prediction
# Comparison of classification methods: decision tree, random forest, XGBoost, kNN and logistic regression

## Project Overview
The aim of this project is to create a classification model, to predict **gym customer churn** (identifying customers who are likely to resign in the upcoming month). The project includes a comparison of 5 classification models: decision tree, random forest, XGBoost, k-nearest neighbors and logistic regression. Then the best model is selected, taking into consideration model's performance on train and test dataset as well as its complexity and interpretability. Finally, SHAP profiles are presented in order to illustrate how the model makes its predictions. 

## Data

### Data Source
Data used in the project comes from the Kaggle website. Link to the dataset: https://www.kaggle.com/datasets/adrianvinueza/gym-customers-features-and-churn/data.  
Dataset involves features such as e.g. demographic data, information if the client lives/works in the proximity of the gym, contract period, average additional charges and information, if client resigned from the gym membership. 

Dataset: Gym Customers Features and Churn by Adrian Vinueza  
Source: Kaggle  
License: CC BY-NC-SA 4.0  

### Data Preprocessing
The dataset used in the analysis exhibited significant class imbalance, included outliers as well as multicollinear variables. Therefore, preprocessing of the dataset involved: feature selection, winsorization and undersampling. 

Altough some of the models were robust to outliers (tree-based models), kNN and logistic regression models are implemented, it was decided that outliers will be brought to the nearest non-outlier values (winsorization), regardless of the model used. 
To balance the dataset, an undersampling technique was used (as the standard SMOTE method cannot be applied on categorical variables). Given that the majority class consists of non-churning customers and the purpose of the model is to identify churning customers, reducing the size of the majority class did not negatively affect the model's performance. 

## Objectives
The purpose of the project was to build a classification model to predict customer churn at the gym. The most important aspect was a successful identification of the potentially resigning customers, so high recall for the churn class was prioritized in the modelling process. Another aim of the analysis was to identify the characteristics of the churning and loyal customers. 

## Methodology
The project compares 5 classification methods: Decision Tree, Random Forest, XGBoost, k-Nearest Neighbors and Logistic Regression. In each case a fine tuning of the parameters was conducted. The metric maximized in the model selection was **recall** - due to the fact that the main purpose of the model is a successful identification of churning customers. Each of the built models allowed for a different degree of interpretability and feature importance analysis. Final model was analyzed using SHAP profiles, both individual and global. 

## Results

### Model Performance and Final Model Selection

<img width="1134" height="372" alt="result_comparison_5" src="https://github.com/user-attachments/assets/1008d63b-de1a-4f81-9bf6-2c1dbc10ad18" />

Each of the trained models exhibited high predictive abilities, both on the train and the test dataset (accuracy and recall values around 0.8-0.9). The best-performing model (based on the test recall metric) was the **logistic regression model**, which allowed to make predictions with **accuracy over 0.85 and recall over 0.95** on both train and test datasets. 

### SHAP Profiles
The logistic regression model expresses the log-odds of the positive outcome as a linear combination of the predictors. Therefore SHAP profiles for this type of model describe additive contribution of each variable on the log-odds of the occurence of positive class. Negative values of log-odds indicate that the probability of the positive class is lower than 50% (predicted class = 0). Positive values, on the contrary, indicate the probability of the positive class greater than 50% (predicted class = 1).

### Global Feature Importance

<img width="845" height="354" alt="shap_summary_profile" src="https://github.com/user-attachments/assets/c917b88c-adce-4a29-8576-58e510cab545" />



Features most significantly impacting the churn prediction turned out to be: 
* **customer's lifetime** - customers with longer lifetime tend to churn less,
* **frequency of class attendance in the current month** - customers who rarely attended classes in the recent time are more likely to churn, 
* **length of the contract period**- short contract periods can be associated with churning customers, 
* **customer's age** - younger customers tend to churn more than the older ones, 
* **time left until the end of the contract period** - customers with expiring contract are more likely to churn.  

Similar conclusions could also be drawn based on the decision tree and random forests models. 
These patterns allow the gym's management to identify the potential churning customers and take appropriate measures to prevent their resignation. 

### Individual SHAP analysis

<img width="2116" height="617" alt="shap_waterfall_plot2" src="https://github.com/user-attachments/assets/4f18187a-9eca-4ccc-9b13-c29a60a11c21" />


The presented SHAP profile shows the result of the model for the customer classified as churning (*churn* = 1). The main variables impacting the result were *lifetime*, *age* and *contract_period*. Due to the lack of customer's lifetime, their young age (27 years old), an only 1-month contract period and infrequent participation in classes in the current month, their chances of churning are high. The amount of customer's additional charges lowers the chances of them churning, but its contribution to the overall prediction is marginal. 


## Technologies
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Sci-Py, imbalanced-learn, SHAP, Jupyter Notebook


## Use of AI
AI tools were used as a supporting resource during the development of this project, mainly for debugging, explaining Python concepts, and improving code readability. 
