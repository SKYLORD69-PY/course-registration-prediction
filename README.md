# 🚀 EduCast AI  
# Automated ML Pipeline for Engineering Workshop Enrollment Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg">
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange.svg">
  <img src="https://img.shields.io/badge/Database-SQLite-green.svg">
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-red.svg">
  <img src="https://img.shields.io/badge/Pipeline-Fully%20Automated-purple.svg">
</p>

---

## 📌 Overview

EduCast AI is a fully automated machine learning pipeline that predicts engineering workshop enrollment for upcoming academic terms.

Unlike traditional ML projects, this system simulates a real-world lifecycle:
- New data is generated continuously
- Models retrain automatically
- Versions are created every run (v1, v2, v3…)
- Dashboard reflects live system state

---

## 🎯 Problem Statement

Universities need to forecast workshop enrollments to:
- Optimize resources
- Plan faculty
- Avoid under/over capacity

Goal:
Predict next-term workshop enrollment using historical academic patterns.

---

## 🚀 Key Features

- Fully Automated ML Pipeline (single command)
- Synthetic data generation
- SQL-based data storage
- Multi-model training (4 models per run)
- Infinite model versioning (v1 → v2 → v3 → …)
- Live dashboard with version comparison
- Continuous retraining system
- Data drift visibility across versions

---

## 🏗️ Project Structure

project/
├── main.py
├── project_config.py
├── database/
├── scripts/
├── models/
├── dashboard/
└── artifacts/

---

## ⚙️ Full Pipeline Flow

main.py →
    generate data →
    store in SQL →
    preprocess →
    train models →
    evaluate →
    save version →
    update active model →
    launch dashboard

---

## 🧪 Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- SQLite  
- Streamlit  
- Joblib  

---

## 🤖 Models Used

- Linear Regression  
- Ridge Regression  
- Random Forest  
- Gradient Boosting  

Each run trains all models and selects the best automatically.

---

## 🔄 Model Versioning

Every time you run:

Run 1 → v1 → 120 rows  
Run 2 → v2 → 240 rows  
Run 3 → v3 → 360 rows  

Each version stores:
- Model file (.joblib)
- Metrics (model_registry.json)

Active version is stored in:
models/active_version.txt

---

## 📊 Dashboard Features

- Dataset Overview (live DB rows)
- Model Performance across versions
- Version Comparison (v1 vs v2 vs v3…)
- Error Heatmaps
- Model Insights
- Prediction Interface

---

## ▶️ Running the Project

Run full pipeline:

python main.py

This will:
- Generate new data
- Train new model version
- Update system
- Launch dashboard automatically

---

Run dashboard separately:

streamlit run dashboard/app.py

---

## 📈 Evaluation Metrics

- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- R² Score  
- MAPE (Mean Absolute Percentage Error)  

---

## 🛠️ Installation

git clone https://github.com/your-username/educast-ai.git  
cd educast-ai  

python -m venv .venv  
.venv\Scripts\activate   # Windows  

pip install -r requirements.txt  

---

## ⚠️ Important Notes

- Do NOT commit:
  - database (.db)
  - model files (.joblib)
  - raw datasets  

- Always run from project root

---

## 💡 Key Learnings

- End-to-end ML pipeline engineering  
- Model lifecycle management  
- SQL data integration  
- Automated retraining systems  
- Version-controlled ML workflows  
- Debugging real-world pipeline issues  

---

## 🚀 Future Improvements

- XGBoost / LightGBM  
- Hyperparameter tuning  
- FastAPI deployment  
- Cloud hosting  
- Real dataset integration  

---

## 📌 Summary

EduCast AI is not just a model — it is a self-evolving ML system that continuously learns from new data and improves over time.

---

## 👨‍💻 Author

Developed as part of an academic ML project focused on production-ready system design.
