# 🦇 Batman Crime Predictor – Fighting Crime with AI  

_"Because even Batman needs data to fight crime"_  

This project applies **Machine Learning** to analyze and predict crime patterns in the **City of Los Angeles**.  
Inspired by Batman’s mission to fight crime, the project uses **real-world crime data** to build predictive models that classify crimes into **violent vs. non-violent** categories.  

---

## 📂 Dataset  

The dataset used in this project is the official **[Crime Data from 2020 to Present – Los Angeles](https://catalog.data.gov/dataset/crime-data-from-2020-to-present)**, provided by the **Los Angeles Police Department (LAPD)**.  

👉 **Only a sample dataset is uploaded here. Please download the full dataset from the official source above if you want to run the complete analysis.**

🚀 **Try the Streamlit App here** [Batman Predictor: Fighting Crime With Ai](https://batman-predictor-fighting-crime-with-ai.streamlit.app/)

---

## ⚙️ Project Workflow  

1. **Data Loading & Cleaning**
   - Handle missing values & duplicated rows.  
   - Drop irrelevant features.  
   - Encode categorical variables.  
   - Feature engineering: Extract time-based features (hour, day, month, year).  

2. **Exploratory Data Analysis (EDA)**
   - Visualizations using **Matplotlib, Seaborn, and Plotly**.  
   - Crime heatmaps, trends, and victim demographics.  
   - Folium map with interactive crime markers.  

3. **Preprocessing**
   - Iterative Imputation for missing values.  
   - Outlier handling.  
   - Feature scaling: StandardScaler, RobustScaler, PowerTransformer, log transform.  

4. **Model Training**
   - Baseline models: Logistic Regression, KNN, Decision Tree, Naive Bayes.  
   - Ensemble models: Random Forest, Extra Trees, Bagging, AdaBoost.  
   - Gradient boosting: XGBoost, LightGBM, CatBoost.  

5. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1.  
   - Confusion Matrices & Classification Reports.  
   - Top features identified using Random Forest.  

---

## 🤖 Models Used  

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Extra Trees Classifier  
- Gaussian Naive Bayes  
- Bagging Classifier  
- AdaBoost  
- XGBoost  
- LightGBM  
- CatBoost  

All trained models are saved as `.pkl` files inside the `Files/` directory.  

---

## 📊 Visualizations  

- **Interactive Folium Map**.  
- **Crime heatmap** by day & hour.  
- **Violin plots** for crime types across hours.  
- **Monthly trends** showing rise/fall in crime rates.  
- **Victim demographics** (age, gender, descent).  
- **Weapon usage distribution**.  

---

## 📈 Results  

| Model                   | Accuracy | Precision | Recall | F1   |
|--------------------------|----------|-----------|--------|------|
| Logistic Regression      | 0.8855   | 0.8831    | 0.8230 | 0.8520 |
| KNN                      | 0.9926   | 0.9901    | 0.9913 | 0.9907 |
| Decision Tree            | 0.9999   | 0.9999    | 0.9999 | 0.9999 |
| Random Forest            | 0.9999   | 0.9999    | 0.9999 | 0.9999 |
| Extra Trees              | 0.9999   | 0.9999    | 0.9998 | 0.9999 |
| Gaussian Naive Bayes     | 0.8697   | 0.8884    | 0.7713 | 0.8257 |
| Bagging                  | 0.9999   | 0.9999    | 0.9999 | 0.9999 |
| AdaBoost                 | 0.9890   | 0.9807    | 0.9920 | 0.9863 |
| XGBoost                  | 0.9999   | 0.9999    | 1.0000 | 0.9999 |
| LightGBM                 | 0.9999   | 0.9999    | 1.0000 | 0.9999 |
| CatBoost                 | 0.9999   | 0.9999    | 0.9999 | 0.9999 |

👉 **Best performers:** XGBoost & LightGBM achieved the highest accuracy (≈ 100%).  
👉 Logistic Regression & Naive Bayes were weaker baselines, while ensemble/boosting methods dominated.  

---
## 🚀 How to Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/Mennaateya/Batman-Predictor-Fighting-Crime-with-AI.git
   cd Batman-Predictor-Fighting-Crime-with-AI
   pip install -r requirements.txt
---
## ⚡ Future Work

- Experimenting with Deep Learning models (RNNs, LSTMs) for time-based crime prediction.
- Adding an alert system for real-time crime hot-spot detection.
- Hyperparameter tuning with Optuna for more robust optimization.

---

## 📝 Contact

👩‍💻 LinkedIn: https://www.linkedin.com/in/menna-ateya

📧 Email: mennaateya30@gmail.com

🦇 Built with AI power — because Gotham deserves better.
