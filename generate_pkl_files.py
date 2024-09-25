import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_and_save_model(dataset_path, forest_name):
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Data Preprocessing: Apply log transformation to the 'area' column
    df['log_area'] = df['area'].apply(lambda x: np.log(x + 1))

    # Select relevant features and target
    features = df[['temp', 'rain', 'RH', 'wind']]
    target = df['log_area']

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

    # Train the RandomForest model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best model after tuning
    best_model = grid_search.best_estimator_

    # Save the trained model and scaler
    model_filename = f'model_{forest_name}.pkl'
    scaler_filename = f'scaler_{forest_name}.pkl'

    with open(model_filename, 'wb') as model_file:
        pickle.dump(best_model, model_file)

    with open(scaler_filename, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    os.makedirs('pkl', exist_ok=True)
    os.rename(model_filename, f'pkl/{model_filename}')
    os.rename(scaler_filename, f'pkl/{scaler_filename}')



    print(f"Model and scaler saved for {forest_name}.")

    # Example usage with different datasets for different forests
forest_datasets = {
    'algeria_forest': './data/algeria_forest.csv',
    'algerian_forest': './data/algerian_forest.csv',
    'sidibel_forest': './data/sidibel_forest.csv'
}

for forest_name, dataset_path in forest_datasets.items():
    print("Training model for:", forest_name)
    print("---------------------------------------")
    train_and_save_model(dataset_path, forest_name)
    print("---------------------------------------")
