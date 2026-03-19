# EcoDrive: Universal Automotive AI & Engineering Suite 🌍🚘⚡

![EcoDrive Registry](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![GenAI](https://img.shields.io/badge/AI-Gemini%202.0%20Flash-vibrant)
![CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-teal)

**EcoDrive** is a specialized high-performance analytics suite and professional engineering assistant. It combines telemetry-based fuel consumption prediction with a world-class **Virtual Chief Engineer** to provide a 360-degree view of vehicle efficiency, engineering history, and future transportation technology.

---

## 👨‍💻 Creator
**Gaurav Akshat Jain**

## 🎯 Key Features

### 🛠️ Virtual Chief Engineer ("engineer help")
- **Gemini 2.0 Flash Powered**: State-of-the-art Generative AI providing deep technical insights on vehicle architecture, thermodynamics, and performance.
- **Specialized Engineering Persona**: Acts as a Virtual Chief Engineer with expertise in ICE mechanics, EV battery management systems, and high-performance telemetry.
- **Intelligent Fallback Heuristics**: When cloud-connectivity is unavailable, the system switches to professional local heuristics for efficiency coaching and vehicle analysis.
- **Standalone Mode**: Includes `engineer_help_bot.py`, a dedicated specialist interface for pure engineering consultation.

### 📚 Universal Vehicle Knowledge Base
- **NHTSA vPIC Integration**: Professional-grade VIN decoding for precise vehicle identification (year, make, model, engine specs, plant location).
- **CarQuery Trim Analysis**: Best-effort enrichment for technical specifications, performance data, and drivetrain configurations.
- **Smart Caching Engine**: Persistent SQLite-based caching for vehicle data to ensure rapid response times and offline availability.
- **Iconic Model Inference**: Advanced parsing to identify iconic models (Mustang, Skyline, 911, etc.) and automatically infer manufacturer data.

### 📊 Telemetry-Based Analytics
- **Machine Learning Predictor**: Uses a `RandomForestRegressor` to analyze real-time telemetry (Speed, RPM, Throttle, Load) and predict fuel consumption (L/100km).
- **Interactive Eco-Optimization**: Live contextual feedback (e.g., gear shift recommendations, throttle management) to minimize carbon footprint.
- **Real-Time Efficiency Monitoring**: Global status alerts (Sustainable vs. High Consumption) based on predictive models.

### 🎨 Modern "Cyber-Eco" UI
- **Premium Design System**: Built with `CustomTkinter` featuring a high-contrast Teal-on-Black aesthetic.
- **Panel-Based Dashboard**: Organized dual-tab layout separating professional AI consulting from live data analytics.
- **Interactive Engineering Status**: Real-time status indicators for Gemini connectivity, quota management, and telemetry stream health.

## ⚙️ Tech Stack
- **Core Engine**: Python 3.10+
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Generative AI**: Google Generative AI (Gemini 2.0 Flash), Python-Dotenv
- **GUI Framework**: CustomTkinter
- **Database**: SQLite3 (Vehicle Data Cache)

## 🚀 Quick Start

### 1. Configure API Access
Create a `.env` file in the root directory and add your Gemini API Key:
```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize the Intelligence
1. **Generate Data**: `python train_model.py` (or provide your own telemetry CSV)
2. **Launch Suite**: `python app.py`
3. **Standalone Engineer Chat**: `python engineer_help_bot.py`

## 🧠 Application Architecture
1. **`app.py`**: The main execution environment. Orchestrates the UI, ML predictions, and GenAI integration.
2. **`vehicle_knowledge.py`**: The core data engine. Handles VIN decoding, external API calls, and local SQLite caching.
3. **`engineer_help_bot.py`**: Dedicated engineering consultant interface.
4. **`train_model.py`**: Training pipeline for the fuel consumption prediction engine.

## 🌍 UN SDG Alignment
- **Goal 11: Sustainable Cities.** Reducing urban vehicular footprint through predictive modeling.
- **Goal 13: Climate Action.** Engineering-driven driver behavior optimization for net-zero carbon output.

---
*Developed with a passion for clean technology & professional automotive engineering.*
