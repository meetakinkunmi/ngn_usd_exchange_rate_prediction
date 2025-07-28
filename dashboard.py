import streamlit as st
import matplotlib.pyplot as plt
from exchange_rate_model.pipeline import full_pipeline

# Page config
st.set_page_config(page_title="USD/NGN", layout="wide")

st.title("📈 Exchange Rate Forecast Dashboard")
st.markdown("Forecasting Weekly USD/NGN Exchange Rate Trends – Stay Ahead of the Curve")

# Sidebar for user input
st.sidebar.header("Configuration")
db_path = st.sidebar.text_input("Database Path", "data/exchange_rate_data.db")
table_name = st.sidebar.text_input("Table Name", "ngn_usd_data")
lags = st.sidebar.slider("Lag Features", min_value=1, max_value=10, value=4)
n_splits = st.sidebar.slider("TimeSeriesSplit (n_splits)", min_value=2, max_value=10, value=2)

if st.sidebar.button("Run Forecast"):
    with st.spinner("Please wait..."):
        fig, metrics_df, y_next, last_df = full_pipeline(db_path, table_name, lags, n_splits)

    # Two-column layout: Plot on the left, metrics + forecast on the right
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("📊 Weekly Exchange Rate Trend: Forecast vs Actual (NGN/USD)")
        st.pyplot(fig)

    with col2:
        st.markdown("🗓️ Next Week Forecast")
        st.metric(label="Forecasted Exchange Rate", value=f"{y_next:.4f}")

        st.markdown("---")
        st.subheader("📉 Evaluation Metrics")
        st.dataframe(metrics_df.style.format(precision=4), use_container_width=True)

    st.markdown("---")
    with st.expander("📂 Last Actual vs Predicted Rate"):
        st.dataframe(last_df.style.format(precision=4), use_container_width=True)

    st.success("✅ Forecast completed.")