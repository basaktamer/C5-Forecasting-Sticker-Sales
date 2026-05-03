---
title: Sales Forecaster - Kaggle Sticker Challenge
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
python_version: "3.10"
license: gpl-3.0
---

# Sticker Sales Prediction Project

## 📌 Project Overview
This project predicts the sales volume (`num_sold`) for various Kaggle-branded products across different stores and countries. It was developed as part of a career transition into **Data Science**, focusing on Time-Series regression and feature engineering.

## 🚀 Technical Highlights
- **Model:** XGBoost Regressor.
- **Metric:** Achieved a **6.42% MAPE** (Mean Absolute Percentage Error) on the validation set.
- **Key Strategy:** Utilized **Log Transformation** ($y = \log(1+x)$) to handle the right-skewed distribution of sales data, which improved the MAPE from 16% to 6.4%.
- **Feature Engineering:** Extracted temporal features such as `day_of_week`, `is_weekend`, and `month` to capture seasonality and the "Weekend Effect."

## 📂 File Structure
- `app.py`: Streamlit application code.
- `xgboost_model.json`: Pre-trained XGBoost model.
- `model_columns.pkl`: List of features used during training (to ensure input alignment).
- `requirements.txt`: Python dependencies.

## 🛠️ Local Setup
To run this project locally with **Python 3.10**:

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate