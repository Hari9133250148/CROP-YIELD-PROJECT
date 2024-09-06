from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the preprocessor and model
with open('dtr.pkl', 'rb') as model_file:
    dtr = pickle.load(model_file)
with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            Year = int(request.form['Year'])
            average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
            pesticides_tonnes = float(request.form['pesticides_tonnes'])
            avg_temp = float(request.form['avg_temp'])
            Area = request.form['Area']
            Item = request.form['Item']

            # Prepare features for prediction
            features_dict = {
                'Year': [Year],
                'average_rain_fall_mm_per_year': [average_rain_fall_mm_per_year],
                'pesticides_tonnes': [pesticides_tonnes],
                'avg_temp': [avg_temp],
                'Area': [Area],
                'Item': [Item]
            }
            features = pd.DataFrame(features_dict)

            # Transform features
            transform_features = preprocessor.transform(features)
            prediction = dtr.predict(transform_features).reshape(1, -1)

            # Render the result
            return render_template('index.html', prediction=prediction[0][0])
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('index.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
