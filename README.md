# EcoDrive: Professional Automotive AI & Analytics Suite 🌍🚘⚡

![EcoDrive Registry](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![GenAI](https://img.shields.io/badge/AI-Gemini%201.5%20Flash-vibrant)
![CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-teal)

**EcoDrive** is a specialized high-performance analytics suite and professional engineering assistant. It combines telemetry-based fuel consumption prediction with a world-class **Professional Automotive AI** to provide a 360-degree view of vehicle efficiency, engineering history, and future transportation technology.

---

## 👨‍💻 Creator
**Gaurav** **Akshat Jain**

## 🎯 Key Features

### 🤖 AutoBot: Professional Engineering Suite
- **Gemini 1.5 Flash Powered**: State-of-the-art Generative AI providing deep technical insights on vehicle architecture, thermodynamics, and performance.
- **Strictly Automotive Focus**: A specialized system prompt ensures the bot remains a professional resource for vehicle-related queries.
- **Universal History & Future**: Deep knowledge spanning from the 1886 Benz Motorwagen to 2050 VTOL taxis and solid-state battery concepts.
- **Hybrid Brain Architecture**: Real-time cloud intelligence with a robust local fallback database for offline technical specs (including chassis codes like AN150, A80, 992).

### 📊 Telemetry-Based Analytics
- **Machine Learning Predictor**: Uses a `RandomForestRegressor` to analyze real-time telemetry (Speed, RPM, Throttle, Load) and predict fuel consumption (L/100km).
- **Interactive Eco-Optimization**: Live contextual feedback (e.g., gear shift recommendations, throttle management) to minimize carbon footprint.
- **Real-Time Efficiency Monitoring**: Global status alerts (Sustainable vs. High Consumption) based on predictive models.

### 🎨 Modern "Cyber-Eco" UI
- **Premium Design System**: Built with `CustomTkinter` featuring a high-contrast Teal-on-Black aesthetic.
- **Panel-Based Dashboard**: Organized dual-tab layout separating professional AI consulting from live data analytics.
- **Interactive Engineering Status**: Live "Scanning" animations and dynamic status headers.

## ⚙️ Tech Stack
- **Core Engine**: Python 3.8+
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Generative AI**: Google Generative AI (Gemini 1.5 Flash), Python-Dotenv
- **GUI Framework**: CustomTkinter (Modernized Tkinter wrapper)

## 🚀 Quick Start

### 1. Configure API Access
Create a `.env` file in the root directory and add your Gemini API Key:
```env
GEMINI_API_KEY=your_api_key_here
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize the Intelligence
1. **Generate Data**: `python generate_mock_data.py`
2. **Train Model**: `python train_model.py`
3. **Launch Suite**: `python app.py`

## 🧠 Application Architecture
1. **`generate_mock_data.py`**: Simulates physics-based telemetry correlations. 
2. **`train_model.py`**: Trains the ensemble Random Forest model on generated telemetry.
3. **`app.py`**: The main execution environment. Initializes the professional AI, loads the ML model, and handles the modern UI event loop.

## 🌍 UN SDG Alignment
- **Goal 11: Sustainable Cities.** Reducing urban vehicular footprint through predictive modeling.
- **Goal 13: Climate Action.** Engineering-driven driver behavior optimization for net-zero carbon output.

---
*Developed with a passion for clean technology & professional automotive engineering.*
