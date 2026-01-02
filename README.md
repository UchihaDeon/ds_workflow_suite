# Data Science Workflow Suite

A modular, end-to-end pipeline for time series analysis, forecasting, and deployment. Built for academic and professional use, this suite integrates preprocessing, EDA, feature engineering, modeling, evaluation, and deployment into a unified framework.

#  Current State

The project is fully functional and includes:

Preprocessing Module: Handles datetime conversion, frequency alignment, and missing value treatment.

EDA Module: Generates summary statistics, correlation matrices, distribution plots, and time series visualizations.

Feature Engineering Module: Supports scaling, encoding, lag features, and rolling averages.

Modeling Module: Unified interface for ML models (Linear Regression, Random Forest), statistical models (ARIMA), and deep learning (LSTM).

Evaluation Module: Computes regression, classification, and forecasting metrics; supports model comparison.

Deployment Module: Offers FastAPI endpoints and Streamlit dashboard templates for serving models.

Test Suite: Pytest-based tests for all modules ensure reliability and maintainability.

# 📦 Project Structure

ds_workflow_suite/
├── app.py
├── config.yaml
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── deployment.py
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_eda.py
│   ├── test_feature_engineering.py
│   ├── test_modeling.py
│   └── test_evaluation.py
└── README.md

# 🛠️ Setup Instructions

# Clone the repo
git clone https://github.com/UchihaDeon/data-science-workflow-suite.git
cd data-science-workflow-suite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Launch API
uvicorn utils.deployment:app --reload

📈 Recent Updates

✅ Added full test coverage for all modules

✅ Integrated FastAPI deployment with prediction endpoint

✅ Generated architecture diagram and flow chart

✅ Improved modular imports with __init__.py files

✅ Streamlined preprocessing and feature engineering functions

📌 Next Steps

[ ] Add Streamlit dashboard for interactive model exploration

[ ] Integrate YAML-based model configuration

[ ] Add support for multivariate forecasting

[ ] Publish documentation site using MkDocs

## 👨‍💻 Author 

Deon — BCA undergraduate, full-stack developer, and data science intern. Passionate about building scalable, user-centric platforms and presenting complex ideas with clarity.

# 📄 License

MIT License. See LICENSE file for details.

# 🙌 Contributions

Feel free to fork, star, and submit pull requests. Feedback and suggestions are welcome!