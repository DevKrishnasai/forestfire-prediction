import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Directory where forest CSV files are stored
CSV_DIR = "./data"

# Dictionary of forest CSV files
forest_csv_files = {
    'Algeria': 'algeria_forest.csv',
    'Algerian': 'algerian_forest.csv',
    'Sidibel': 'sidibel_forest.csv'
}

# Dictionary of total area for each forest (in hectares)
forest_total_areas = {
    'Algeria': 124,
    'Algerian': 300,
    'Sidibel': 200,
}

@app.route('/')
def home():
    # Provide the list of available forests
    forests = list(forest_csv_files.keys())
    return render_template('index.html', forests=forests)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        forest_selected = request.form['forest']
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        RH = float(request.form['RH'])
        wind = float(request.form['wind'])

        # Load the corresponding model and scaler
        model_filename = f'./pkl/model_{forest_selected.lower() + "_forest"}.pkl'
        scaler_filename = f'./pkl/scaler_{forest_selected.lower() + "_forest"}.pkl'

        # Ensure the model files exist
        if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
            return render_template(
                'index.html',
                forests=list(forest_csv_files.keys()),
                prediction_text='Model or scaler file not found. Please check the files.'
            )

        model = pickle.load(open(model_filename, 'rb'))
        scaler = pickle.load(open(scaler_filename, 'rb'))

        # Scale the inputs
        features = np.array([[temp, rain, RH, wind]])
        scaled_features = scaler.transform(features)

        # Predict the burnt area
        pred_log_area = model.predict(scaled_features)
        pred_burnt_area = np.exp(pred_log_area) - 1  # Convert from log scale

        # Get the total area for the selected forest
        total_area = forest_total_areas[forest_selected]

        # Return the result to the webpage
        return render_template(
            'index.html',
            forests=list(forest_csv_files.keys()),
            prediction_text=f'Predicted Burnt Area for {forest_selected}: {pred_burnt_area[0]:.2f} hectares out of {total_area} hectares'
        )

if __name__ == '__main__':
    app.run(debug=True)
