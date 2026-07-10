# IBM Telco Customer Churn Prediction & A/B Testing Dashboard

An end-to-end data science project that analyses the IBM Telco Customer Churn dataset (~7,000 customers) to understand why customers leave, test potential interventions, and predict churn. The Streamlit dashboard exposes five interactive tabs covering exploratory data analysis, SQL-based insights, statistical hypothesis tests, a simulated A/B retention experiment, and two machine-learning models (Logistic Regression and Random Forest) evaluated with ROC-AUC.

## Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.9+ |
| Dashboard | Streamlit |
| Visualisation | Plotly |
| Statistics | SciPy, statsmodels |
| Machine Learning | scikit-learn |
| Data | pandas, NumPy, SQLite |

## Screenshot

*Run the app and take a screenshot to replace this placeholder.*

## How to Run Locally

```bash
# 1. Clone / download the project
cd churn-project

# 2. Place the dataset
#    Download "WA_Fn-UseC_-Telco-Customer-Churn.csv" from Kaggle
#    and save it as data/telco_churn.csv

# 3. Install dependencies
pip install -r requirements.txt

# 4. Load data into SQLite
python load_db.py

# 5. Launch the dashboard
streamlit run app.py
```

## Project Structure

```
churn-project/
├── data/
│   └── telco_churn.csv      # IBM Telco dataset (user-supplied)
├── load_db.py                # CSV → SQLite loader
├── app.py                    # Streamlit dashboard (5 tabs)
├── churn.db                  # Generated SQLite database
├── requirements.txt          # Pinned Python dependencies
├── progress.txt              # Build log
├── desc.txt                  # Algorithm descriptions
└── README.md
```

## Deployment

The app can be deployed on any platform that supports Streamlit:

1. **Streamlit Community Cloud** — push the repo to GitHub, connect via share.streamlit.io, set `data/telco_churn.csv` as a data source.
2. **Docker** — wrap in a Dockerfile that copies the data, installs requirements, and runs `streamlit run app.py --server.port 8501`.
3. **Any VM / PaaS** — `pip install -r requirements.txt && python load_db.py && streamlit run app.py`.
