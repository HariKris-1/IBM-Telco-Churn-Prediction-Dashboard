"""Customer Churn Prediction & A/B Testing Dashboard."""

import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

DB_PATH = "churn.db"

# ── Streamlit page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction & A/B Testing",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction & A/B Testing Dashboard")


# ── Data helpers ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    return df


@st.cache_data
def run_sql(query: str):
    conn = sqlite3.connect(DB_PATH)
    result = pd.read_sql(query, conn)
    conn.close()
    return result


# ── Load once ───────────────────────────────────────────────────────────
df = load_data()

# ── Tabs ────────────────────────────────────────────────────────────────
tab_eda, tab_sql, tab_hyp, tab_ab, tab_ml = st.tabs(
    ["EDA", "SQL Insights", "Hypothesis Tests", "A/B Test", "Churn Model"]
)

# =====================================================================
# TAB 1 — Exploratory Data Analysis
# =====================================================================
with tab_eda:
    st.header("Exploratory Data Analysis")

    total_customers = len(df)
    churn_rate = df["Churn_binary"].mean()

    col1, col2 = st.columns(2)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Overall Churn Rate", f"{churn_rate:.2%}")

    st.subheader("Tenure Distribution by Churn Status")
    fig_tenure = px.histogram(
        df,
        x="tenure",
        color="Churn",
        barmode="overlay",
        nbins=30,
        labels={"tenure": "Tenure (months)", "count": "Customers"},
        color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
        opacity=0.7,
    )
    fig_tenure.update_layout(bargap=0.05)
    st.plotly_chart(fig_tenure, width="stretch")

    st.subheader("Monthly Charges Distribution by Churn Status")
    fig_charges = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        labels={"MonthlyCharges": "Monthly Charges ($)"},
        color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
    )
    st.plotly_chart(fig_charges, width="stretch")

# =====================================================================
# TAB 2 — SQL Insights
# =====================================================================
with tab_sql:
    st.header("SQL Analysis (queries run against churn.db)")

    # ── Interactive SQL Explorer ────────────────────────────────────────
    st.subheader("🔍 Interactive SQL Explorer")
    st.info(
        "Type any SQL query below and click **Run Query** to execute it against "
        "`churn.db`. The table is called `customers`."
    )

    # Show table schema for reference
    with st.expander("📋 View Table Schema (column names & types)"):
        schema_df = run_sql("PRAGMA table_info(customers);")
        st.dataframe(
            schema_df[["name", "type"]].rename(
                columns={"name": "Column", "type": "SQLite Type"}
            ),
            width="stretch",
            hide_index=True,
        )

    user_query = st.text_area(
        "Enter your SQL query:",
        value="SELECT * FROM customers LIMIT 10;",
        height=120,
        key="sql_explorer",
    )

    if st.button("▶️ Run Query", type="primary"):
        if user_query.strip():
            try:
                result_df = run_sql(user_query)
                st.success(f"Query returned **{len(result_df):,}** row(s).")
                st.dataframe(result_df, width="stretch")
            except Exception as e:
                st.error(f"**SQL Error:** {e}")
        else:
            st.warning("Please enter a SQL query first.")

    st.divider()
    st.subheader("📊 Pre-Built SQL Insights")

    # Query 1: Churn rate by contract type
    st.subheader("Churn Rate by Contract Type")
    q1 = """
    SELECT Contract,
           COUNT(*) AS total_customers,
           SUM(Churn_binary) AS churned,
           ROUND(AVG(Churn_binary) * 100, 2) AS churn_rate_pct
    FROM customers
    GROUP BY Contract
    ORDER BY churn_rate_pct DESC;
    """
    st.code(q1, language="sql")
    st.dataframe(run_sql(q1), width="stretch")

    # Query 2: Tenure quartiles with NTILE
    st.subheader("Tenure Quartiles (NTILE window function)")
    q2 = """
    SELECT tenure_quartile,
           MIN(tenure) AS min_tenure,
           MAX(tenure) AS max_tenure,
           COUNT(*) AS customer_count,
           ROUND(AVG(Churn_binary) * 100, 2) AS churn_rate_pct
    FROM (
        SELECT tenure, Churn_binary,
               NTILE(4) OVER (ORDER BY tenure) AS tenure_quartile
        FROM customers
    )
    GROUP BY tenure_quartile
    ORDER BY tenure_quartile;
    """
    st.code(q2, language="sql")
    st.dataframe(run_sql(q2), width="stretch")

    # Query 3: Cumulative churn count
    st.subheader("Cumulative Churn Count by Tenure (window function)")
    q3 = """
    SELECT tenure,
           Churn_binary,
           SUM(Churn_binary) OVER (ORDER BY tenure
                                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
               AS cumulative_churn
    FROM customers
    ORDER BY tenure
    LIMIT 200;
    """
    st.code(q3, language="sql")
    cum_df = run_sql(q3)
    st.dataframe(cum_df, width="stretch")

    fig_cum = px.line(
        cum_df, x="tenure", y="cumulative_churn",
        labels={"tenure": "Tenure (months)", "cumulative_churn": "Cumulative Churn Count"},
    )
    st.plotly_chart(fig_cum, width="stretch")

# =====================================================================
# TAB 3 — Hypothesis Tests
# =====================================================================
with tab_hyp:
    st.header("Hypothesis Tests")

    # Chi-square: Contract type vs Churn
    st.subheader("Chi-Square Test: Contract Type vs Churn")
    contingency = pd.crosstab(df["Contract"], df["Churn"])
    chi2, p_chi, dof, _ = chi2_contingency(contingency)

    st.write(f"**Chi-square statistic:** {chi2:.4f}")
    st.write(f"**p-value:** {p_chi:.4e}")
    if p_chi < 0.05:
        st.success(
            "The relationship between Contract type and Churn is **statistically "
            "significant** (p < 0.05). Contract type is associated with churn."
        )
    else:
        st.info(
            "No statistically significant relationship between Contract type and "
            "Churn at the 0.05 level."
        )

    st.divider()

    # T-test: Monthly Charges
    st.subheader("Independent T-Test: Monthly Charges (Churned vs Non-Churned)")
    churned_charges = df.loc[df["Churn_binary"] == 1, "MonthlyCharges"]
    retained_charges = df.loc[df["Churn_binary"] == 0, "MonthlyCharges"]
    t_stat, p_t = ttest_ind(churned_charges, retained_charges)

    st.write(f"**t-statistic:** {t_stat:.4f}")
    st.write(f"**p-value:** {p_t:.4e}")
    if p_t < 0.05:
        st.success(
            "There is a **statistically significant** difference in Monthly Charges "
            "between churned and non-churned customers (p < 0.05)."
        )
    else:
        st.info(
            "No statistically significant difference in Monthly Charges between "
            "the two groups at the 0.05 level."
        )

# =====================================================================
# TAB 4 — A/B Test Simulation
# =====================================================================
with tab_ab:
    st.header("A/B Test Simulation")
    st.warning(
        "⚠️ **This section is a simulated experiment, not real A/B test data.** "
        "The retention outcomes below are artificially generated to demonstrate "
        "the statistical testing workflow."
    )

    rng = np.random.RandomState(42)
    churned_df = df[df["Churn_binary"] == 1].copy()
    n_churned = len(churned_df)
    indices = rng.permutation(n_churned)
    half = n_churned // 2

    # Simulate binary retention outcomes
    control_retained = rng.binomial(1, 0.05, size=half)
    treatment_retained = rng.binomial(1, 0.15, size=half)

    control_rate = control_retained.mean()
    treatment_rate = treatment_retained.mean()

    col1, col2 = st.columns(2)
    col1.metric("Control Retention Rate", f"{control_rate:.2%}", help="Baseline — no offer")
    col2.metric("Treatment Retention Rate", f"{treatment_rate:.2%}", help="Retention offer applied")

    st.subheader("Two-Proportion Z-Test")
    count = np.array([control_retained.sum(), treatment_retained.sum()])
    nobs = np.array([half, half])
    z_stat, p_z = proportions_ztest(count, nobs, alternative="two-sided")

    st.write(f"**z-statistic:** {z_stat:.4f}")
    st.write(f"**p-value:** {p_z:.4e}")
    if p_z < 0.05:
        st.success(
            "The difference in retention rates between control and treatment groups "
            "is **statistically significant** (p < 0.05). The simulated retention "
            "offer appears to have an effect."
        )
    else:
        st.info("No statistically significant difference detected at the 0.05 level.")

    st.subheader("Sample Size Estimation (Power Analysis)")
    effect_size = proportion_effectsize(0.05, 0.15)
    power_analysis = NormalIndPower()
    required_n = power_analysis.solve_power(
        effect_size=effect_size, alpha=0.05, power=0.80, alternative="two-sided"
    )

    st.write(f"**Effect size (Cohen's h):** {effect_size:.4f}")
    st.write(f"**Required sample size per group:** {int(np.ceil(required_n)):,}")
    st.write(
        f"To detect the difference between a 5% and 15% retention rate with 80% "
        f"power at α = 0.05, you would need at least **{int(np.ceil(required_n)):,}** "
        f"customers in each group."
    )

# =====================================================================
# TAB 5 — Churn Model
# =====================================================================
with tab_ml:
    st.header("Churn Prediction Models")

    # Prepare features
    drop_cols = ["customerID", "Churn", "Churn_binary"]
    feature_df = df.drop(columns=drop_cols)
    feature_df = pd.get_dummies(feature_df, drop_first=True)
    X = feature_df.values
    y = df["Churn_binary"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Logistic Regression (scaled)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)

    st.subheader("ROC-AUC Scores")
    col1, col2 = st.columns(2)
    col1.metric("Logistic Regression", f"{lr_auc:.4f}")
    col2.metric("Random Forest", f"{rf_auc:.4f}")

    # Feature importances
    st.subheader("Top 10 Feature Importances (Random Forest)")
    importances = pd.Series(rf.feature_importances_, index=feature_df.columns)
    top10 = importances.nlargest(10).sort_values()

    fig_imp = px.bar(
        x=top10.values,
        y=top10.index,
        orientation="h",
        labels={"x": "Importance", "y": "Feature"},
    )
    fig_imp.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_imp, width="stretch")
