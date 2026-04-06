# 🛠️ SmartMaint AI — Predictive Maintenance System

SmartMaint AI is an end-to-end Machine Learning-based Predictive Maintenance System designed to forecast equipment failures before they occur. The system processes industrial data, trains predictive models, generates explainable insights using SHAP, and deploys predictions through a Flask-based web interface.

This project demonstrates a complete production-style ML workflow including data generation, preprocessing, model training, explainability, visualization, and web deployment.

---

# 🚀 Features

✅ End-to-End Machine Learning Pipeline
✅ Predictive Maintenance Failure Detection
✅ Data Preprocessing & Feature Engineering
✅ Automated Model Training
✅ SHAP-Based Model Explainability
✅ Visualization of Feature Importance
✅ Flask Web Interface for Predictions
✅ Structured Data Pipeline
✅ Report and Plot Generation

---

# 🧠 Problem Statement

Industrial equipment failures often lead to unexpected downtime, financial loss, and safety risks. Traditional maintenance approaches rely on fixed schedules rather than real-time predictions.

SmartMaint AI solves this problem using machine learning models that analyze equipment sensor data to predict potential failures before they happen, enabling proactive maintenance planning.

---

# 🏗️ System Architecture

```
Raw Data
   │
   ▼
Data Generation (generate_data.py)
   │
   ▼
Data Preprocessing (preprocess.py)
   │
   ▼
Model Training (model_train.py)
   │
   ▼
Model Saving → data/models/
   │
   ▼
SHAP Explainability (shap_explain.py)
   │
   ▼
Outputs → Plots & Reports
   │
   ▼
Flask Web API (app.py)
   │
   ▼
Web Interface (HTML/CSS/JS)
```

---

# 📁 Project Structure

```
SmartMaint_AI/

├── api/                          # Flask Web Application
│   ├── app.py                    # Main Flask server
│   │
│   ├── templates/                # HTML UI templates
│   │   └── index.html
│   │
│   └── static/                   # Frontend assets
│       ├── script.js
│       └── style.css
│
├── data/                         # Dataset storage
│   ├── raw/                      # Original data
│   ├── processed/                # Cleaned data
│   └── models/                   # Saved ML models
│
├── notebooks/                    # Jupyter experiments
│
├── outputs/                      # Generated outputs
│   ├── plots/
│   │   └── dependence/
│   └── reports/
│
├── src/                          # Core ML logic
│   ├── preprocess.py
│   ├── model_train.py
│   ├── shap_explain.py
│   └── utils.py
│
├── generate_data.py              # Synthetic data generator
├── run_pipeline.py               # Complete ML pipeline runner
│
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

---

# ⚙️ Installation & Setup

## Step 1 — Clone Repository

```
git clone https://github.com/creator1404/final_megaProject.git
cd final_megaProject
```

---

## Step 2 — Create Virtual Environment

Windows:

```
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3 — Install Dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Run Full ML Pipeline

This will:

* Generate data
* Preprocess data
* Train model
* Generate explainability plots

```
python run_pipeline.py
```

---

## Start Flask Web Application

```
cd api
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

# 📊 Model Explainability

This project uses **SHAP (SHapley Additive exPlanations)** to interpret model predictions.

Generated visualizations include:

* Feature Importance Plots
* SHAP Summary Plots
* SHAP Dependence Plots

These visualizations help understand how different features influence equipment failure predictions.

---

# 📈 Outputs Generated

After running the pipeline:

```
outputs/
├── plots/
│   └── dependence/
├── reports/
```

Includes:

* Model performance metrics
* Feature importance charts
* SHAP explanation graphs

---

# 🧪 Technologies Used

## Programming

* Python
* JavaScript
* HTML
* CSS

## Machine Learning

* Scikit-learn
* SHAP
* NumPy
* Pandas

## Visualization

* Matplotlib
* Seaborn

## Web Framework

* Flask

---

# 📊 Use Cases

SmartMaint AI can be applied in:

🏭 Manufacturing Industries
⚙️ Industrial Equipment Monitoring
🚂 Transportation Systems
⚡ Power Plants
🏢 Smart Infrastructure

---

# 🎯 Future Improvements

* Add real-time sensor data streaming
* Deploy using Docker
* Add cloud deployment
* Integrate dashboard analytics
* Implement deep learning models
* Add REST API endpoints
* Add user authentication
---

# 📜 License

This project is intended for educational and demonstration purposes.

---
