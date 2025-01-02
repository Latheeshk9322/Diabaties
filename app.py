from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained KNN model
with open("model/knn_model.pkl", "rb") as model_file:
    knn_model = pickle.load(model_file)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['dpf'])
        age = float(request.form['age'])

        # Prepare input for the model
        user_input = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree_function, age
        ]])

        # Predict the outcome
        outcome = knn_model.predict(user_input)[0]

        # Return the result
        result = "Diabetic" if outcome == 1 else "Non-Diabetic"
        return render_template('index.html', prediction_text=f'The predicted outcome is: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'An error occurred: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
