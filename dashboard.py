import streamlit as st
import pandas as pd
from exchange_rate_forecast.pipeline import predict_next_exchange_rate

st.set_page_config(page_title="₦/$ Exchange Rate Predictor", layout="centered")

st.title("📈 Naira to Dollar Exchange Rate Forecast")
st.markdown("Use this app to forecast the next ₦/$ rate based on recent historical data.")

uploaded_file = st.file_uploader("Upload a single-row CSV of lagged & logged data", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file, index_col=0)

    st.subheader("Input Features")
    st.dataframe(input_df)

    if st.button("🔮 Predict Exchange Rate"):
        prediction = predict_next_exchange_rate(input_df)
        st.success(f"Predicted Next Exchange Rate: ₦{prediction:,.2f}")