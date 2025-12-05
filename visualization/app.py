import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))       
ROOT_DIR = os.path.dirname(BASE_DIR)                         
DATA_DIR = os.path.join(ROOT_DIR, "user_data")               
MODEL_DIR = os.path.join(ROOT_DIR, "user_data")            

def introduction():
    st.markdown(
                """
## Second-Hand Car Price Prediction

### Introduction
Develop and optimize a regression model capable of accurately predicting the transaction prices of used cars using both the **explicit features** (e.g., power, kilometer, body type) and the **anonymous features** $(v_0 - v_{14})$, while minimizing the MAE score on the test sets.
""")
@st.cache_data
def load_dataset():
    df_path = os.path.join(DATA_DIR, "processed_features.csv")   
    df = pd.read_csv(df_path)
    return df

@st.cache_resource
def load_models():
    rf_path= os.path.join(MODEL_DIR, "rf_best.joblib")   
    xgb_path= os.path.join(MODEL_DIR, "XGBoost_Model.joblib")   
    rf_model=joblib.load(rf_path)
    xgb_model=joblib.load(xgb_path)
    return rf_model, xgb_model


@st.cache_data
def compute_predictions(df, _rf_model, _xgb_model):
    X=df.drop(columns=["price_log"])
    y_true=np.expm1(df["price_log"].values)
    y_rf=np.expm1(_rf_model.predict(X))
    y_xgb=np.expm1(_xgb_model.predict(X))
    out=pd.DataFrame({
        "SaleID": df["SaleID"].values,
        "true_price": y_true,
        "rf_pred": y_rf,
        "xgb_pred": y_xgb
    })
    return out


def display_features(df, index):
    st.subheader("Car Features")
    row = df.iloc[index]
    col_feat1,col_val1,col_feat2,col_val2=st.columns([2.5,1.5, 2.5,1.5])
    for i, col in enumerate(df.columns):
        if col == "price_log":
            continue
        value = row[col]
        if i%2==0:
            with col_feat1:
                st.info(col)
            with col_val1:
                st.success(str(value))
        else:
            with col_feat2:
                st.info(col)
            with col_val2:
                st.success(str(value))


def display_predictions_for_car(df_pred_all, saleid):
    row =df_pred_all[df_pred_all["SaleID"]==saleid].iloc[0]
    true_price=float(row["true_price"])
    pred_rf=float(row["rf_pred"])
    pred_xgb=float(row["xgb_pred"])
    err_rf =pred_rf-true_price
    err_xgb= pred_xgb-true_price
    pct_rf= 100* err_rf /true_price
    pct_xgb=100 *err_xgb/true_price
    col_rf, col_xgb, col_real = st.columns(3)

    with col_real:
        st.subheader("Real Price")
        st.success(f"{true_price:,.0f} €")
    with col_rf:
        st.subheader("Random Forest Prediction")
        st.info(f"{pred_rf:,.0f} €")
        st.write(f"Error: {err_rf:,.0f} € ({pct_rf:+.1f}%)")
    with col_xgb:
        st.subheader("XGBoost Prediction")
        st.info(f"{pred_xgb:,.0f} €")
        st.write(f"Error: {err_xgb:,.0f} € ({pct_xgb:+.1f}%)")
    st.subheader("Price Comparison (Visualization)")
    chart_df = pd.DataFrame({
        "Model": ["True price", "Random Forest", "XGBoost"],
        "Price": [true_price, pred_rf, pred_xgb]
    }).set_index("Model")
    st.bar_chart(chart_df)

def display_global_scatter(df_pred_all):
    st.subheader("Overall Model Performance (True vs Predicted)")
    fig, ax = plt.subplots()
    ax.scatter(df_pred_all["true_price"],df_pred_all["rf_pred"],alpha=0.3,label="Random Forest")
    ax.scatter(df_pred_all["true_price"],df_pred_all["xgb_pred"],alpha=0.3,label="XGBoost")
    low = df_pred_all["true_price"].min()
    high = df_pred_all["true_price"].max()
    ax.plot([low, high], [low, high], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("True price (€)")
    ax.set_ylabel("Predicted price (€)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)

def eda_section(df, rf_model):
    st.header("Second-Hand Car Price Prediction-Exploratory Data Analysis (EDA)")
    st.markdown("This section gives an overview of the training dataset used by the models.")
    st.subheader("1. Dataset Overview")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Number of rows",df.shape[0])
    col_b.metric("Number of columns",df.shape[1])
    col_c.metric("Number of features",df.shape[1] - 2) 
    st.write("Preview of dataset:")
    st.dataframe(df.head())
###
    st.subheader("2. Price Distribution")
    price = np.expm1(df["price_log"])
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    sns.histplot(price, bins=50, ax=ax[0], color="skyblue")
    ax[0].set_title("Raw Price Distribution")
    sns.histplot(df["price_log"], bins=50, ax=ax[1], color="orange")
    ax[1].set_title("Log Price Distribution (price_log)")
    st.pyplot(fig)

###
    st.subheader("3. Key Feature Distributions")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.histplot(df["power"], bins=40, ax=ax2, color="green")
    ax2.set_title("Power Distribution")
    st.pyplot(fig2)
###
    st.subheader("4.Most Common Models and Brands")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Top 15 Models")
        st.bar_chart(df["model"].value_counts().head(15))
    with col2:
        st.write("Top 15 Brands")
        st.bar_chart(df["brand"].value_counts().head(15))
    st.subheader("5. Correlations")
    selected_cols = ["price_log", "power", "kilometer", "model", "brand"]
    corr=df[selected_cols].corr()
    fig3,ax3 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax3)
    st.pyplot(fig3)
        
def main():
    df = load_dataset()
    rf_model, xgb_model = load_models()
    df_pred_all = compute_predictions(df, rf_model, xgb_model)
    introduction()
    eda_section(df, rf_model)
    display_global_scatter(df_pred_all)

    st.markdown("---")
    st.set_page_config(page_title="Used Car - Model Comparison", layout="wide")
    st.title("Second-Hand Car Price Prediction - Model Comparison")

    st.markdown(
        """
This application lets you select a **car by its `SaleID`** and compare:

- The **real price**
- The **Random Forest prediction**
- The **XGBoost prediction**
- The absolute and percentage errors for both models
        """
    )
    df =load_dataset()
    rf_model,xgb_model=load_models()
    df_pred_all = compute_predictions(df,rf_model,xgb_model)
    st.sidebar.header("Choose a car")
    saleid_lst =sorted(df["SaleID"].unique().tolist())
    saleid = st.sidebar.selectbox("SaleID", options=saleid_lst)
    index = df.index[df["SaleID"] == saleid][0]
    st.write(f"### Selected SaleID: {saleid}")
    display_predictions_for_car(df_pred_all, saleid)
    display_features(df, index)

if __name__ == "__main__":
    main()


