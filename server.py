import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import os

app = Flask(__name__)

CSV_DIR = "./data"
forest_csv_files = {
    'Algeria': 'algeria_forest.csv',
    'Algerian': 'algerian_forest.csv',
    'Sidibel': 'sidibel_forest.csv'
}

forest_total_areas = {
    'Algeria': 13720,
    'Algerian': 14920,
    'Sidibel': 78900,
}

@app.route('/')
def home():
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

        model_filename = f'./pkl/model_{forest_selected.lower()}_forest.pkl'
        scaler_filename = f'./pkl/scaler_{forest_selected.lower()}_forest.pkl'

        if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
            return jsonify({'error': 'Model or scaler file not found. Please check the files.'})

        model = pickle.load(open(model_filename, 'rb'))
        scaler = pickle.load(open(scaler_filename, 'rb'))

        features = np.array([[temp, rain, RH, wind]])
        scaled_features = scaler.transform(features)

        pred_log_area = model.predict(scaled_features)
        pred_burnt_area = np.exp(pred_log_area) - 1

        total_area = forest_total_areas[forest_selected]
        percentage_burnt = (pred_burnt_area[0] / total_area) * 100

        return jsonify({
            'forest': forest_selected,
            'burnt_area': round(pred_burnt_area[0], 2),
            'total_area': total_area,
            'percentage_burnt': round(percentage_burnt, 2)
        })

@app.route('/forest_data', methods=['GET'])
def forest_data():
    forest = request.args.get('forest')
    if forest not in forest_csv_files:
        return jsonify({'error': 'Forest not found'})

    csv_path = os.path.join(CSV_DIR, forest_csv_files[forest])
    df = pd.read_csv(csv_path)

    return jsonify({
        'avg_temp': round(df['temp'].mean(), 2),
        'avg_rain': round(df['rain'].mean(), 2),
        'avg_RH': round(df['RH'].mean(), 2),
        'avg_wind': round(df['wind'].mean(), 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
