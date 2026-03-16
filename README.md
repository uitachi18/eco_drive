# EcoDrive: Telemetry-Based Emission Optimizer 🌍🚘

![EcoDrive](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-lightgrey)

**EcoDrive** is a telemetry-based predictive modeling application designed to analyze high-performance vehicle telemetry and predict fuel consumption rates in real-time. By providing actionable insights into driver behavior, EcoDrive actively aligns with **UN Sustainable Development Goals (SDGs) 11 & 13** by promoting sustainable, efficient transportation and climate action.

---

## 👨‍💻 Creator
**Gaurav** **Akshat Jain**

## 🎯 Key Features
- **Synthetic Telemetry Simulation**: Generates high-fidelity mock data mimicking F1-style or track telemetry (Speed, RPM, Throttle, Engine Load).
- **Machine Learning Architecture**: Utilizes a highly robust `RandomForestRegressor` for nonlinear predictions of `Fuel Consumption (L/100km)`.
- **Modern User Interface**: A sleek, dark-themed glassmorphism GUI built natively with Tkinter that requires minimal overhead.
- **Dynamic Recommendation Engine**: Evaluates inputs live and provides contextual feedback (e.g., "Shift to a higher gear to reduce RPM," "Ease off the accelerator to save fuel").
- **Real-Time UI Warnings**: Triggers warning states when driving parameters yield unsustainably high carbon emissions/fuel consumption.

## ⚙️ Tech Stack
- **Language**: Python 3.x
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Model Serialization**: Joblib
- **Frontend GUI**: Tkinter, ttk

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/uitachi18/eco_drive.git
cd eco_drive
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Telemetry Data
Run the mock generation engine to create 1000 rows of training telemetry data.
```bash
python generate_mock_data.py
```

### 4. Train the ML Model
Execute the training pipeline to generate the `.pkl` payload.
```bash
python train_model.py
```
*(You will see metrics like Mean Squared Error and R2 Score logged directly to your terminal).*

### 5. Launch the Application
Start the UI dashboard.
```bash
python app.py
```

## 🧠 Application Architecture
1. **`generate_mock_data.py`**: Utilizes complex physics distributions to heavily correlate RPM, Load, and Speed metrics against expected fuel consumption. 
2. **`train_model.py`**: The ingest pipeline splits synthetic data 80/20, feeding it into an ensemble Random Forest learner to establish high-variance feature correlations.
3. **`app.py`**: The driver application. Instantiates the model directly into a singleton-like execution environment and evaluates the user matrix asynchronously.

## 🌍 UN SDG Alignment
- **Goal 11: Sustainable Cities and Communities.** By actively optimizing telemetry, EcoDrive helps reduce urban vehicular footprint.
- **Goal 13: Climate Action.** The recommendation engine actively aids in altering driver behavior to reduce immediate CO2 output.

---
*Developed with a passion for clean technology & machine learning.*
