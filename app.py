from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('heart_disease_model.joblib')

# Initialize Flask app
app = Flask(__name__)

# Home Route - Display the form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route - Process the form data and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user input from form
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            chest_pain = int(request.form['chest_pain'])
            bp = int(request.form['bp'])
            cholesterol = int(request.form['cholesterol'])
            fbs = int(request.form['fbs'])
            ekg = int(request.form['ekg'])
            max_hr = int(request.form['max_hr'])
            exercise_angina = int(request.form['exercise_angina'])
            st_depression = float(request.form['st_depression'])
            slope = int(request.form['slope'])
            vessels = int(request.form['vessels'])
            thallium = int(request.form['thallium'])

            # Prepare input data for prediction
            input_data = np.array([[age, sex, chest_pain, bp, cholesterol, fbs, ekg,
                                    max_hr, exercise_angina, st_depression, slope,
                                    vessels, thallium]])

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Output result
            result = "Risk of Heart Disease: Presence (High Risk)" if prediction == 1 else "Risk of Heart Disease: Absence (Low Risk)"
        
        except ValueError:
            result = "Invalid input. Please enter correct values."

    return render_template('result.html', prediction=result)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
