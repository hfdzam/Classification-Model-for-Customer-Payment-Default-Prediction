# Classification Model for Customer Payment Default Prediction

## Overview

In the banking industry, accurately predicting which customers are likely to subscribe to a term deposit is crucial for maximizing revenue and customer retention. However, identifying potential subscribers (the 'y' target) has become increasingly challenging in recent years. One of the main risks is the occurrence of false negatives—failing to identify interested customers—which can result in lost revenue and decreased customer satisfaction. This notebook aims to enhance predictive models to better detect customers who are likely to default on their payments, thereby helping banks manage credit risk more effectively and reduce potential losses.

## Notebook Purpose

In this notebook, you will:

### Data Collection:
- Acquire the customer data from banking databases.
- Store the data in a structured format for further analysis.

### Data Preprocessing:
- Clean and preprocess the dataset, handling missing values and encoding categorical variables.
- Address data imbalance using techniques such as SMOTE to ensure the model can effectively detect potential customers.

### Exploratory Data Analysis (EDA):
- Conduct an exploratory analysis to understand the distribution of the target variable and the influence of various features.
- Visualize the relationships between features like `duration`, `pdays`, `previous`, and customer subscription status using interactive charts.
- Highlight key insights, such as the imbalance in the target variable where only 11.5% of customers subscribe to term deposits.

### Model Development:
- Train and compare multiple classification models including SVM, K-Nearest Neighbors, Decision Tree, Random Forest, and AdaBoost.
- Evaluate models based on recall, particularly focusing on reducing false negatives (FN) as detecting potential subscribers is critical for maximizing revenue.
- Apply hyperparameter tuning to optimize the models, with a focus on improving the recall score by at least 10%.

### Model Evaluation and Selection:
- Evaluate the performance of each model, particularly focusing on recall for the positive class.
- Select the best-performing model based on its ability to accurately predict potential subscribers.

### Model Limitations:
- Discuss the limitations of the models, including challenges in predicting customer behavior and the impact of data imbalance.
- Analyze the trade-offs between different models, noting how SMOTE improved recall by 15-20%.

### Conclusion:
- Summarize the findings, emphasizing the effectiveness of the best-performing model in predicting potential customers.
- Provide recommendations for using the model in a real-world setting, such as filtering data upfront to minimize incorrect predictions and optimize processing time.

### Deployment:
- Deploy the final model on Hugging Face, making it accessible for real-time predictions and further fine-tuning.
- Provide a link to the deployed model for easy access and testing.


For details and to view the notebook, check out [this Jupyter Notebook](P1M2_Hafidz_Masruri.ipynb).

For inference, refer to [this Jupyter Notebook](P1M2_Hafidz_Masruri_inf.ipynb).

For deployment, visit this [Link](https://huggingface.co/spaces/hfdzam/Prediction_M2).

## Tools and Libraries Used

- **Pandas**: For data manipulation and analysis, especially handling tabular data.
- **NumPy**: For numerical operations and handling arrays.
- **Scikit-Learn**: For building and evaluating machine learning models, including classification algorithms and SMOTE.
- **Matplotlib** and **Seaborn**: For creating static, animated, and interactive visualizations.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: For addressing data imbalance by generating synthetic samples.
- **Jupyter Notebook**: For writing code in an interactive environment and documenting the workflow.
- **Hugging Face**: For deploying the final model and making it accessible for real-time predictions.
