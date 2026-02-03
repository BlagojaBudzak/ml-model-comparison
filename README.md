# ML Model Comparison

This repository contains implementations of **regression, classification, and clustering models** in Python, with visualizations and performance evaluation. The goal is to provide a clean comparison of different machine learning models on small datasets.

## Project Structure

```
ml-model-comparison/
├── data/
│ ├── Mall_Customers.csv
│ ├── Salary_Data.csv
│ └── Social_Network_Ads.csv
├── notebooks/
│ ├── classification_models.ipynb
│ ├── clustering_models.ipynb
│ └── regression_models.ipynb
├── src/
│ ├── classification_models.py
│ ├── clustering_models.py
│ └── regression_models.py
├── README.md
└── requirements.txt
```

- `data/` – CSV datasets for each type of model  
- `notebooks/` – Jupyter notebooks for exploration and plotting  
- `src/` – Python scripts for running models directly  
- `README.md` – project description and instructions  
- `requirements.txt` – Python dependencies  

## Datasets

1. **Social_Network_Ads.csv** – Used for classification models (target: Purchased or not)  
2. **Salary_Data.csv** – Used for regression models (predict salary from experience)  
3. **Mall_Customers.csv** – Used for clustering models (Annual Income vs Spending Score)  

## Features

- **Classification**
  - Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest  
  - Accuracy calculation, confusion matrix, and decision boundary visualization  

- **Regression**
  - Linear Regression, Polynomial Regression, Support Vector Regression, Decision Tree, Random Forest  
  - Mean Squared Error (MSE), R² score, and prediction plots  

- **Clustering**
  - Hierarchical Clustering & K-Means  
  - Dendrogram, cluster scatter plots, and cluster statistics  

## Usage

# 1. Clone the repository:

```bash
git clone https://github.com/BlagojaBudzak/ml-model-comparison.git
```

# 2. Install dependencies:
pip install -r requirements.txt

# 3. Run scripts:

## Classification
python src/classification_models.py

## Regression
python src/regression_models.py

## Clustering
python src/clustering_models.py


# Developed by Blagoja Budzakoski
