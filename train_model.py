import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 1: Load the Dataset
data = pd.read_csv('dataset.csv')

# Step 2: Data Preprocessing
# Assuming the target column is named 'Heart Disease' with values 'Presence' and 'Absence'
data['Heart Disease'] = data['Heart Disease'].apply(lambda x: 1 if x == 'Presence' else 0)

# Separate features and target variable
X = data.drop(columns=['Heart Disease'])
y = data['Heart Disease']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 6: Save the Trained Model
joblib.dump(model, 'heart_disease_model.joblib')
print("\nModel saved as 'heart_disease_model.joblib'")
