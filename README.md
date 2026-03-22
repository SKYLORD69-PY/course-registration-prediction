# 📊 EduCast AI  
### Engineering Workshop Enrollment Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg">
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange.svg">
  <img src="https://img.shields.io/badge/Database-SQLite-green.svg">
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-red.svg">
  <img src="https://img.shields.io/badge/Models-Versioned%20v1%20%7C%20v2-purple.svg">
</p>

---

## 📌 Overview

**EduCast AI** is an end-to-end machine learning system designed to predict **engineering workshop enrollment for upcoming academic terms**.

It simulates a real-world ML workflow including data generation, database storage, model training, versioning, and dashboard-based insights.

---

## 🎯 Problem Statement

Universities need to plan workshop capacity in advance, but enrollment varies based on:

- Academic term  
- School  
- Year of study  
- Credit load  
- Compulsory course rules  

> **How many students will enroll in the Engineering Workshop in the next term?**

---

## 🚀 Key Features

- End-to-End ML Pipeline  
- Domain-aware synthetic dataset  
- Multiple ML models  
- Model versioning (v1, v2)  
- Continuous retraining  
- Interactive dashboard  

---

## 🏗️ Project Structure

```bash
project/
├── main.py
├── project_config.py
├── data/
├── database/
├── scripts/
├── models/
├── dashboard/
└── artifacts/
```

---

## ⚙️ How It Works

1. Generate synthetic academic data  
2. Store data in SQLite  
3. Preprocess and engineer features  
4. Train ML models  
5. Evaluate performance  
6. Save best model (versioning)  
7. Visualize via dashboard  

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

---

## 🔄 Model Versioning

- v1 → baseline  
- v2 → improved  

Active version stored in:

```bash
models/active_version.txt
```

---

## 📊 Dashboard Features

- Overview  
- Insights  
- Predictor  
- Model Performance  
- Model Registry  
- Data Health  
- V1 vs V2 comparison  

---

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/educast-ai.git
cd educast-ai
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
```

### 3. Activate Environment

#### Windows
```bash
.venv\Scripts\activate
```

#### macOS / Linux
```bash
source .venv/bin/activate
```

### 4. Upgrade pip
```bash
pip install --upgrade pip
```

### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Run Full Pipeline
```bash
python main.py
```

### Run Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📈 Evaluation Metrics

- MAE  
- RMSE  
- R² Score  
- MAPE  

---

## 🔁 Example Workflow

```bash
# Run pipeline
python main.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## ⚠️ Troubleshooting

### Import Errors
```bash
# Make sure you're in project root
cd educast-ai
```

### Dashboard Not Launching
```bash
streamlit run dashboard/app.py
```

### Reinstall Dependencies
```bash
pip install -r requirements.txt --force-reinstall
```

---

## 💡 Key Learnings

- End-to-end ML system design  
- Data engineering  
- Model versioning  
- Dashboard development  
- Debugging real-world ML issues  

---

## 🚀 Future Improvements

- XGBoost / LightGBM  
- Hyperparameter tuning  
- FastAPI deployment  
- Cloud hosting  
- Real dataset integration  

---

## 📌 Summary

**EduCast AI** is a complete machine learning system combining data engineering, model building, and visualization for academic decision-making.

---

## 👨‍💻 Author

Developed as part of an academic ML project focused on real-world system design.
