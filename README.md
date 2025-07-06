📈 NGN/USD Exchange Rate Forecasting
This project is focused on building a time series forecasting model to predict the Nigerian Naira to US Dollar (₦/$) exchange rate using historical weekly data. It is designed with a modular Python package architecture to support end-to-end analysis, modeling, and deployment.

🔍 Objectives
Understand the temporal behavior of the NGN/USD exchange rate.

Apply time series modeling using machine learning algorithms.

Develop a reusable, production-ready forecasting pipeline.

Generate accurate, interpretable forecasts to support trade, investment, and economic policy planning.

🧰 Tools & Technologies
Python, pandas, NumPy

scikit-learn

matplotlib, seaborn

SQLite (remote/local data)

Git & GitHub

🚀 Key Features
Modular Python package structure for preprocessing, modeling, and evaluation.

Log transformation and lag-based feature engineering to handle non-stationarity and temporal dependencies.

Baseline linear regression model with performance evaluation (MAE, RMSE, R²).

Time-aware train/test split to preserve data chronology.

Visual comparison of actual vs predicted exchange rate values.

Ready for deployment and integration into automated systems.

📦 Structure
text
Copy
Edit
exchange_rate_model/
│
├── exchange_rate_model/          # Core package (functions)
├── notebooks/                    # Exploratory notebooks
├── data/                         # Local data files (excluded from Git)
├── tests/                        # Unit tests
├── setup.py, requirements.txt    # Packaging and dependencies
├── README.md                     # Project summary and instructions
└── LICENSE                       # License file