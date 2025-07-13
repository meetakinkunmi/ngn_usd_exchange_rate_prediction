from setuptools import setup, find_packages

setup(
    name="exchange_rate_forecast",
    version="0.1.0",
    author="Akinkunmi",
    author_email="ioakinkunmi@gmail.com", 
    description="Forecasting Naira/USD exchange rate using ensemble machine learning models",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/meetakinkunmi/ngn_usd_exchange_rate_prediction",  
    project_urls={
        "Bug Tracker": "https://github.com/meetakinkunmi/ngn_usd_exchange_rate_prediction/issues",
        "Documentation": "https://github.com/meetakinkunmi/ngn_usd_exchange_rate_prediction#readme"
    },
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.10.0",
        "streamlit>=1.18.0",
        "joblib>=1.1.0",
        "statsmodels>=0.13.0"
    ],
    python_requires='>=3.8',
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",               
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="forecasting exchange-rate finance machine-learning ensemble naira dollar",
)