import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def main():
    data_file = 'telemetry_data.csv'
    model_file = 'ecodrive_model.pkl'
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run generate_mock_data.py first.")
        return

    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    # Features and target
    X = df[['Speed', 'Engine_RPM', 'Throttle_Position', 'Engine_Load']]
    y = df['Fuel_Consumption_Rate']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save model
    joblib.dump(model, model_file)
    print(f"Model saved successfully to {model_file}")

if __name__ == '__main__':
    main()
