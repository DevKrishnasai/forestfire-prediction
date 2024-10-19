import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set paths for data and pkl directories
DATA_DIR = './data'
PKL_DIR = './pkl'
os.makedirs(PKL_DIR, exist_ok=True)

# List of forest datasets with filenames in the data directory
forest_datasets = {
    'Nainital': 'nainital_forest.csv',
    'Jim Corbett': 'jim_forest.csv',
    'Bandipur': 'bandipur_forest.csv'
}

# Loop through each forest and generate corresponding model and scaler
for forest_name, filename in forest_datasets.items():
    # Construct the full path to the CSV file
    file_path = os.path.join(DATA_DIR, filename)

    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        continue

    # Select features and target
    X = df[['temp', 'rain', 'RH', 'wind']]
    y = np.log(df['area'] + 1)  # Log(1 + area) to stabilize variance

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train_scaled, y_train)

    # Save the model and scaler as .pkl files
    model_filename = f'{PKL_DIR}/model_{forest_name.lower()}_forest.pkl'
    scaler_filename = f'{PKL_DIR}/scaler_{forest_name.lower()}_forest.pkl'

    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)

    print(f'Model and scaler saved for {forest_name}:')
    print(f'  Model -> {model_filename}')
    print(f'  Scaler -> {scaler_filename}\n')
