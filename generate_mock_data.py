import pandas as pd
import numpy as np
import os

def generate_data(num_samples=1000, output_file='telemetry_data.csv'):
    np.random.seed(42)
    
    # Time in seconds
    time = np.arange(num_samples)
    
    # High-performance simulation
    # Speed (km/h): 50 to 350
    speed = np.random.uniform(50, 350, num_samples)
    
    # Engine RPM: 3000 to 12000 for high performance
    rpm = np.random.uniform(3000, 12000, num_samples)
    
    # Throttle Position (%): 0 to 100
    throttle = np.random.uniform(0, 100, num_samples)
    
    # Engine Load (%): 0 to 100
    load = np.random.uniform(0, 100, num_samples)
    
    # Fuel Consumption calculation (L/100km)
    # Base consumption
    base_consumption = 15.0
    
    # Higher RPM -> dramatically higher consumption
    rpm_factor = (rpm / 12000) ** 2 * 30
    
    # Heavy throttle -> higher consumption
    throttle_factor = (throttle / 100) ** 1.5 * 25
    
    # Speed and load also contribute
    speed_factor = (speed / 350) * 10
    load_factor = (load / 100) * 15
    
    # Add some random noise
    noise = np.random.normal(0, 2, num_samples)
    
    fuel_consumption = base_consumption + rpm_factor + throttle_factor + speed_factor + load_factor + noise
    
    # Ensure it's not negative (though mathematically unlikely here)
    fuel_consumption = np.maximum(fuel_consumption, 5.0)

    df = pd.DataFrame({
        'Time': time,
        'Speed': np.round(speed, 2),
        'Engine_RPM': np.round(rpm, 2),
        'Throttle_Position': np.round(throttle, 2),
        'Engine_Load': np.round(load, 2),
        'Fuel_Consumption_Rate': np.round(fuel_consumption, 2)
    })
    
    df.to_csv(output_file, index=False)
    print(f"Generated {num_samples} rows of high-performance telemetry data and saved to {output_file}")

if __name__ == '__main__':
    generate_data()
