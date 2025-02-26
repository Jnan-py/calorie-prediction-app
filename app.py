import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Set Streamlit page configuration
st.set_page_config(page_title="Calorie Prediction App", layout="wide", page_icon= ":fire")

# Load dataset
DATA_PATH = "Dataset\calories_burnt.csv"
MODEL_PATH = "calories_model.pkl"
SCALER_PATH = "scaler.pkl"

def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

def train_and_save_model():
    st.write("Training model... Please wait.")
    df.replace({'male': 0, 'female': 1}, inplace=True)
    features = df.drop(['User_ID', 'Calories'], axis=1)
    target = df['Calories'].values
    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    models = [LinearRegression(), XGBRegressor(), Lasso(), RandomForestRegressor(), Ridge()]
    best_model, best_error = None, float('inf')
    
    for model in models:
        model.fit(X_train, Y_train)
        val_preds = model.predict(X_val)
        error = mae(Y_val, val_preds)
        if error < best_error:
            best_error, best_model = error, model
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    st.success("Model training complete! Model saved locally.")
    return best_model, scaler

if "model" not in st.session_state:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            st.session_state["model"] = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            st.session_state["scaler"] = pickle.load(f)
    else:
        st.session_state["model"], st.session_state["scaler"] = train_and_save_model()

def generate_visualizations():
    plots = {}
    
    fig1 = px.scatter(df, x='Height', y='Weight', title="Height vs Weight")
    plots["Height vs Weight"] = fig1
    
    features_ = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

    for feature in features_:
        fig = px.scatter(df.sample(1000), x=feature, y='Calories', title=f"{feature} vs Calories")
        plots[f"{feature} vs Calories"] = fig
    
    df["Gender"] = df["Gender"].apply(lambda x: 0 if x=="male" else 1)
    
    fig2 = px.imshow(df.corr(), text_auto=True, title="Feature Correlation Heatmap")
    plots["Correlation Heatmap"] = fig2
    
    return plots

if "plots" not in st.session_state:
    st.session_state["plots"] = generate_visualizations()

st.title("🔥 Calorie Prediction & Visualization App")

tab1, tab2 = st.tabs(["🔮 Prediction","📊 Visualizations"])

with tab2:
    st.header("📊 Data Visualizations")
    for plot_title, fig in st.session_state["plots"].items():
        st.subheader(plot_title)
        st.plotly_chart(fig, width = 350)

with tab1:
    st.header("🔮 Predict Calories Burned")
    
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.radio("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    duration = st.number_input("Exercise Duration (minutes)", min_value=1, max_value=180, value=30)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=80)
    temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
    
    gender = 0 if gender == "Male" else 1
    
    input_features = np.array([[age, gender, height, weight, duration, heart_rate, temp]])
    input_features = st.session_state["scaler"].transform(input_features)
    
    if st.button("Predict Calories"):
        prediction = st.session_state["model"].predict(input_features)[0]
        st.success(f"🔥 Estimated Calories Burned: **{prediction:.2f} kcal**")
