import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import datetime

# 1. MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Kaggle Sticker Sales Forecast", layout="centered")

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    # Load the XGBoost model
    model = xgb.XGBRegressor()
    model.load_model("model.ubj") 
    
    # Load the exact column list from your training phase
    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
        
    return model, model_columns

# Load model and columns
model, model_columns = load_assets()

# --- INTERFACE ---
st.title("📊 Sticker Sales Forecasting")
st.write("Predict sales volume based on historical training data patterns.")

# --- INPUT SECTION ---
with st.sidebar:
    st.header("Input Features")
    date = st.date_input("Select Date", datetime.date(2026, 5, 4))
    
    # Categories based on your data summary
    country = st.selectbox("Country", ["Finland", "Italy", "Singapore", "Norway", "Canada", "Kenya"])
    store = st.selectbox("Store", ["Premium Sticker Mart", "Stickers for Less", "Discount Stickers"])
    product = st.selectbox("Product", ["Kaggle", "Kaggle Tiers", "Kerneler Dark Mode", "Kerneler", "Holographic Goose"])

# --- PREDICTION LOGIC ---
if st.button("Predict Units Sold"):
    # 1. Feature Engineering (Match your training steps)
    input_date = pd.to_datetime(date)
    data = {
        'year': input_date.year,
        'month': input_date.month,
        'day': input_date.day,
        'dayofweek': input_date.dayofweek,
        'is_weekend': 1 if input_date.dayofweek >= 5 else 0
    }
    
    # 2. Build the full feature row using the loaded columns
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # 3. Fill Time Features
    for key, value in data.items():
        if key in input_df.columns:
            input_df.at[0, key] = value
            
    # 4. Fill One-Hot Features
    for feat in [f"country_{country}", f"store_{store}", f"product_{product}"]:
        if feat in input_df.columns:
            input_df.at[0, feat] = 1

    # 5. Model Inference
    raw_pred = model.predict(input_df)
    
    # Reverse log transformation if used during training (np.expm1)
    final_prediction = np.expm1(raw_pred)[0]

    # --- RESULTS ---
    st.markdown("---")
    if final_prediction < 0: final_prediction = 0
    st.metric(label="Predicted Units Sold", value=f"{int(final_prediction)}")
    
    if final_prediction > 0:
        st.success("Calculation complete.")
    else:
        st.warning("The model returned a near-zero prediction. Check feature alignment.")