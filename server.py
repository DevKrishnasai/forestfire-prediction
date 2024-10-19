import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import os
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Load directories from environment variables (or default paths)
CSV_DIR = os.getenv('CSV_DIR', './data')
PKL_DIR = os.getenv('PKL_DIR', './pkl')

# Forest datasets and total areas
forest_csv_files = {
    'Nainital': 'nainital_forest.csv',
    'Jim Corbett': 'jim_forest.csv',
    'Bandipur': 'bandipur_forest.csv'
}

forest_total_areas = {
    'Nainital': 27000,
    'Jim Corbett': 52100,
    'Bandipur': 87400,
}

@app.route('/')
def home():
    forests = list(forest_csv_files.keys())
    return render_template('index.html', forests=forests)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        forest_selected = request.form['forest']
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        RH = float(request.form['RH'])
        wind = float(request.form['wind'])

        # Build model and scaler paths
        model_filename = os.path.join(PKL_DIR, f'model_{forest_selected.lower()}_forest.pkl')
        scaler_filename = os.path.join(PKL_DIR, f'scaler_{forest_selected.lower()}_forest.pkl')

        # Check if model and scaler files exist
        if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
            app.logger.error(f'Model or scaler file not found for {forest_selected}')
            return jsonify({'error': 'Model or scaler file not found. Please check the files.'}), 404

        # Load model and scaler
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_filename, 'rb') as f:
            scaler = pickle.load(f)

        # Prepare input data for prediction
        features = np.array([[temp, rain, RH, wind]])
        scaled_features = scaler.transform(features)

        # Predict burnt area
        pred_log_area = model.predict(scaled_features)
        pred_burnt_area = np.exp(pred_log_area) - 1

        # Calculate percentage of burnt area
        total_area = forest_total_areas[forest_selected]
        percentage_burnt = (pred_burnt_area[0] / total_area) * 100

        # Return the prediction as JSON
        return jsonify({
            'forest': forest_selected,
            'burnt_area': round(pred_burnt_area[0], 2),
            'total_area': total_area,
            'percentage_burnt': round(percentage_burnt, 2)
        })

    except Exception as e:
        app.logger.error(f'Error in predict: {str(e)}')
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

@app.route('/forest_data', methods=['GET'])
def forest_data():
    forest = request.args.get('forest')

    # Check if the forest exists in the dataset mapping
    if forest not in forest_csv_files:
        app.logger.warning(f'Forest not found: {forest}')
        return jsonify({'error': 'Forest not found'}), 404

    # Load the CSV data for the requested forest
    csv_path = os.path.join(CSV_DIR, forest_csv_files[forest])
    df = pd.read_csv(csv_path)

    # Calculate and return average statistics
    return jsonify({
        'avg_temp': round(df['temp'].mean(), 2),
        'avg_rain': round(df['rain'].mean(), 2),
        'avg_RH': round(df['RH'].mean(), 2),
        'avg_wind': round(df['wind'].mean(), 2)
    })

if __name__ == '__main__':
    # Run the app in production mode
    app.run(host='0.0.0.0', port=8000, debug=False)
