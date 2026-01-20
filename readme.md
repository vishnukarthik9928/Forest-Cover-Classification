# 🌲 Forest Cover Classification using Machine Learning

An end-to-end **Machine Learning classification project** that predicts the **forest cover type** of a geographical area using cartographic and environmental features.  
The project includes **data preprocessing, feature engineering, class imbalance handling, model training, and a Streamlit web application** for real-time prediction.

---

## 📌 Project Overview

Forest cover type classification plays a vital role in:
- Environmental monitoring
- Forestry management
- Land-use planning
- Ecological research

This project uses a **Random Forest Classifier** with proper **categorical encoding and class balancing** to accurately predict forest cover types such as *Spruce/Fir, Aspen, Lodgepole Pine*, etc.

---

## 📊 Dataset Information

- **Dataset**: Forest Cover Type Dataset  
- **Rows**: ~145,000  
- **Target Variable**: `Cover_Type`  
- **Classes**:
  - Spruce/Fir  
  - Lodgepole Pine  
  - Ponderosa Pine  
  - Cottonwood/Willow  
  - Aspen  
  - Douglas-fir  
  - Krummholz  

### 🔑 Key Features
- Elevation
- Aspect
- Slope
- Distances to hydrology, roads, fire points
- Hillshade values
- Soil Type (categorical)
- Wilderness Area (categorical)

---

## 🛠️ Tech Stack

- **Python 3.10**
- **Pandas, NumPy**
- **Scikit-learn**
- **Imbalanced-learn (SMOTE)**
- **Streamlit**
- **Joblib**

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Collection
- Loaded dataset using Pandas
- Inspected shape, columns, and target distribution

### 2️⃣ Data Cleaning
- Removed duplicates
- Handled missing values using median imputation

### 3️⃣ Feature Engineering
- Created derived feature:


### 4️⃣ Proper Categorical Encoding (Critical Fix)
- Applied **One-Hot Encoding** for:
- `Soil_Type`
- `Wilderness_Area`
- Prevented incorrect ordinal interpretation of categorical values

### 5️⃣ Class Imbalance Handling
- Used **SMOTE** to balance minority classes (especially Aspen)

### 6️⃣ Model Training
- Algorithm: **Random Forest Classifier**
- Improvements:
- `class_weight="balanced"`
- Tuned tree depth and estimators
- Evaluation:
- Accuracy
- Confusion Matrix
- Classification Report

### 7️⃣ Model Persistence
Saved artifacts:
- `best_model.pkl`
- `preprocessor.pkl`

---

## 🌐 Streamlit Web Application

An interactive web app built using **Streamlit** that allows users to:
- Input terrain and environmental features
- Predict forest cover type in real time
- Correctly identify minority classes like **Aspen**

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
python -m pip install pandas numpy scikit-learn imbalanced-learn streamlit joblib

