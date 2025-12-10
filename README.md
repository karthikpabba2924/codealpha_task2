# codealpha_task2
car-price-prediction : Machine learning model for predicting car prices using regression &amp; feature engineering techniques
🚗 Car Price Prediction using Machine Learning

Predicting used car prices using Python, Machine Learning, and real-world feature engineering.
This project includes preprocessing, feature engineering, multiple ML models, evaluation, and visualization.

📌 Project Overview

This project builds an end-to-end Car Price Prediction System using a real dataset.
It applies multiple regression algorithms to understand which model performs best for predicting a car’s selling price.

🔍 Key Features

📥 Load & clean raw car dataset

⚙️ Handle missing values, scaling & encoding

🚗 Feature engineering (car age, kms per year, brand extraction)

🤖 Multiple ML models trained:

Linear Regression, Ridge, Lasso

Random Forest, Extra Trees

Gradient Boosting, HistGradientBoosting

SVR, KNN

Neural Network (MLPRegressor)

📊 Classification model: Low / Medium / High price category

📈 Visualization: Actual vs Predicted price chart

🧪 Evaluation metrics: RMSE, MAE, R², Accuracy

💾 Easy-to-run Notebook / Python script

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-Learn

Matplotlib

Jupyter Notebook

📂 Dataset

The dataset contains car attributes such as:

Car Name

Year

Present Price

Selling Price

Driven Kilometers

Fuel Type

Selling Type

Transmission

Owner Count

You can replace it with your own car dataset as needed.

🚀 How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/yourusername/car-price-ml.git
cd car-price-ml

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run Jupyter Notebook
jupyter notebook


Open the notebook and execute all cells.

4️⃣ OR run Python script
python car_price_prediction.py

📊 Model Evaluation

The project reports metrics such as:

RMSE – Root Mean Squared Error

MAE – Mean Absolute Error

R² Score – Goodness of fit

Accuracy (for classification)

Models with higher R² and lower RMSE perform the best.

📈 Visualization

Example chart output:

Scatter plot comparing Actual vs Predicted selling prices using RandomForestRegressor

Helps understand model accuracy visually

🧠 Learning Outcomes

Building an end-to-end ML workflow

Understanding preprocessing for structured datasets

Feature engineering for real-world auto datasets

Comparing different ML algorithms

Evaluating model performance with standard metrics
