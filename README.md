# Heart Disease Risk Assessment - Golden Level Project

## Overview
This project is a **Heart Disease Prediction Web Application** is a golden-level project assigned to me as part of my Data Science Internship at **CodeClause** IT Company. This is a web-based machine learning project built using Python and Flask. It employs a machine learning model trained on a Kaggle dataset to predict the risk of heart disease based on user input. The web application features a user-friendly interface, allowing users to input medical parameters and receive a prediction regarding their heart disease risk.

I would like to express my gratitude to **CodeClause** IT Company for providing me with this wonderful opportunity to enhance my skills and knowledge during the internship.


## Features
- **User-Friendly Interface**: A web-based form where users can input medical data such as age, blood pressure, cholesterol levels, etc.
- **Machine Learning Model**: A pre-trained model is used to make predictions about the presence or absence of heart disease.
- **Interactive Results**: Displays a clear result indicating whether the user is at high risk or low risk for heart disease.

## Tech Stack
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS (via Flask's templating)
- **Machine Learning**: Model trained using a dataset from Kaggle and stored as `heart_disease_model.joblib`.

## Prerequisites
- Python 3.x installed on your machine.
- Required Python packages:
  - Flask
  - joblib
  - numpy

## Dataset
The dataset for training the model was sourced from Kaggle. It includes medical data such as age, cholesterol levels, and other relevant features.

### Key Features:
1. **Model Development**: A Random Forest Classifier was trained on the Kaggle dataset to predict heart disease risk.
2. **Web Interface**: A Flask-based UI for users to enter their medical data and view the prediction results.
3. **End-to-End Workflow**: From data preprocessing and model training to deployment in a web application.
4. **Interactive Predictions**: Provides immediate and clear predictions: *High Risk* or *Low Risk* of heart disease.

---

## Dataset
- **Source**: Kaggle ([Dataset Link](https://www.kaggle.com/datasets/kapoorprakhar/cardio-health-risk-assessment-dataset))
- **Details**: The dataset contains medical parameters, including:
  - Age, blood pressure, cholesterol levels, and other heart-health-related features.
  - Target variable: *Heart Disease* (Presence/Absence).

---

## Workflow

### 1. **Model Training**
- **Dataset Loading**: Load the CSV data into a Pandas DataFrame.
- **Preprocessing**: 
  - Encode target labels: *Presence* = 1, *Absence* = 0.
  - Separate features (`X`) and target (`y`).
- **Train-Test Split**:
  - Data split: 80% training, 20% testing.
- **Model Training**:
  - Random Forest Classifier with 100 estimators.
  - Achieved **high accuracy** (output logged during training).
- **Model Evaluation**:
  - **Metrics**: Accuracy, classification report, confusion matrix.
  - Validation results are printed for interpretation.
- **Model Saving**: Trained model is saved as `heart_disease_model.joblib` for deployment.

### 2. **Web Application**
- Built with **Flask**.
- User inputs are collected via an HTML form and passed to the model for prediction.
- Results are dynamically displayed on the results page.

---

## Installation and Setup

### Prerequisites:
- Python 3.x
- Required packages (install via `pip install -r requirements.txt`):
  - Flask
  - joblib
  - pandas
  - scikit-learn
  - numpy

## Output
The model takes 'Age', 'Sex', 'Chest Pain Type', 'Blood Pressure', 'Cholestrol', 'FBS', 'EKG Results', 'Max Heart Rate', 'ST Depression', 'Slope of ST', 'Number of Vessels' and 'Thallium' as input from the user and predicts the output as 'Risk of Heart Disease: Presence (High Risk)' or 'Risk of Heart Disease: Absence (Low Risk)'.

## Acknowledgments
Dataset is collected from Kaggle.
This project is assigned by CodeClause IT Company as a part of 'Data Science Internship'. 
