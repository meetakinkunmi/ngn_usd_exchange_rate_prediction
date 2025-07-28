import streamlit as st
import matplotlib.pyplot as plt
import tempfile
from exchange_rate_model.pipeline import full_pipeline

# Page config
st.set_page_config(page_title="USD/NGN Exchange Rate Forecast", layout="wide")

st.title("📈 Exchange Rate Forecast Dashboard")
st.markdown("Forecasting Weekly USD/NGN Exchange Rate Trends – Stay Ahead of the Curve")

# Sidebar for user input
st.sidebar.header("Configuration")

# File uploader for SQLite .db
uploaded_file = st.sidebar.file_uploader("📂 Upload your SQLite `.db` file", type=["db"])

# Input fields
table_name = st.sidebar.text_input("Table Name", value="ngn_usd_data")
lags = st.sidebar.slider("Lag Features", min_value=1, max_value=10, value=4)
n_splits = st.sidebar.slider("TimeSeriesSplit (n_splits)", min_value=2, max_value=10, value=2)

# Only show forecast button if file is uploaded
if uploaded_file and st.sidebar.button("Run Forecast"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_db_path = tmp_file.name

    with st.spinner("⏳ Running forecast pipeline..."):
        try:
            fig, metrics_df, y_next, last_df = full_pipeline(temp_db_path, table_name, lags, n_splits)

            # Two-column layout
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(
                    '<h3 style="font-size:16px;">📊 Weekly Exchange Rate Trend: Forecast vs Actual (NGN/USD)</h3>',
                    unsafe_allow_html=True
                )
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

            st.success("✅ Forecast completed successfully.")

        except Exception as e:
            st.error(f"❌ An error occurred during processing: {str(e)}")

else:
    st.info("👈 Please upload a `.db` file and click 'Run Forecast'.")