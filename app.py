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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
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
tab_eda, tab_sql, tab_hyp, tab_ab, tab_ml, tab_predict = st.tabs(
    ["EDA", "SQL Insights", "Hypothesis Tests", "A/B Test", "Churn Model",
     "Predict Customer Churn"]
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

    # ── Advanced SQL Analysis ──────────────────────────────────────────
    st.divider()
    st.subheader("🚀 Advanced SQL Analysis")

    # 1. Customer Segmentation by Tenure
    st.subheader("1. Customer Segmentation")
    q4_sql = """
    SELECT 
        CASE 
            WHEN tenure <= 12 THEN 'New Customers (<= 1 Year)'
            WHEN tenure <= 36 THEN 'Regular Customers (1-3 Years)'
            ELSE 'Loyal Customers (> 3 Years)'
        END AS Segment,
        COUNT(*) AS CustomerCount,
        ROUND(AVG(MonthlyCharges), 2) AS AvgMonthlyCharges,
        ROUND(AVG(Churn_binary) * 100, 2) AS ChurnRate
    FROM customers
    GROUP BY Segment
    ORDER BY MIN(tenure);
    """
    st.code(q4_sql, language="sql")
    st.dataframe(run_sql(q4_sql), width="stretch")
    st.info("💡 **Insight:** New customers have significantly higher churn rates compared to loyal customers. As tenure increases, churn rate dramatically drops, highlighting the importance of early retention strategies.")

    # 2. Revenue Analysis by Contract Type
    st.subheader("2. Revenue Analysis by Contract Type")
    q5_sql = """
    SELECT 
        Contract,
        COUNT(*) AS TotalCustomers,
        ROUND(SUM(TotalCharges), 2) AS TotalRevenue,
        ROUND(AVG(TotalCharges), 2) AS AverageRevenue,
        ROUND(AVG(Churn_binary) * 100, 2) AS ChurnRate
    FROM customers
    GROUP BY Contract
    ORDER BY TotalRevenue DESC;
    """
    st.code(q5_sql, language="sql")
    st.dataframe(run_sql(q5_sql), width="stretch")
    st.info("💡 **Insight:** Two-year contracts generate the most total revenue and have the lowest churn rate. Month-to-month contracts have high customer volume but suffer from high churn and lower average revenue per user.")

    # 3. Churn by Payment Method
    st.subheader("3. Churn by Payment Method")
    q6_sql = """
    SELECT 
        PaymentMethod,
        COUNT(*) AS TotalCustomers,
        ROUND(AVG(Churn_binary) * 100, 2) AS ChurnRate
    FROM customers
    GROUP BY PaymentMethod
    ORDER BY ChurnRate DESC;
    """
    st.code(q6_sql, language="sql")
    st.dataframe(run_sql(q6_sql), width="stretch")
    st.info("💡 **Insight:** Customers paying via Electronic Check churn at an alarmingly higher rate (over 45%) compared to other automated or mailed methods.")

    # 4. Highest Revenue Customers (Top 10)
    st.subheader("4. Highest Revenue Customers")
    q7_sql = """
    SELECT 
        customerID,
        Contract,
        tenure,
        ROUND(MonthlyCharges, 2) AS MonthlyCharges,
        ROUND(TotalCharges, 2) AS TotalCharges
    FROM customers
    ORDER BY TotalCharges DESC
    LIMIT 10;
    """
    st.code(q7_sql, language="sql")
    st.dataframe(run_sql(q7_sql), width="stretch")
    st.info("💡 **Insight:** The top 10 customers by revenue mostly have long tenures (71-72 months) and high monthly charges, primarily on Two-year or One-year contracts.")

    # 5. CASE WHEN Analysis (Charges vs Churn)
    st.subheader("5. CASE WHEN Analysis")
    q8_sql = """
    SELECT 
        CASE 
            WHEN MonthlyCharges < 40 THEN 'Low Charges (< $40)'
            WHEN MonthlyCharges < 80 THEN 'Medium Charges ($40 - $80)'
            ELSE 'High Charges (>= $80)'
        END AS ChargeLevel,
        COUNT(*) AS CustomerCount,
        ROUND(AVG(Churn_binary) * 100, 2) AS AverageChurnRate
    FROM customers
    GROUP BY ChargeLevel
    ORDER BY MIN(MonthlyCharges);
    """
    st.code(q8_sql, language="sql")
    st.dataframe(run_sql(q8_sql), width="stretch")
    st.info("💡 **Insight:** Customers with higher monthly bills (>= $80) experience higher average churn rates (~34%) compared to customers with lower monthly bills (< $40).")

    # 6. CTE Analysis (Contract Revenue vs Overall Average)
    st.subheader("6. CTE Analysis")
    q9_sql = """
    WITH ContractAvg AS (
        SELECT 
            Contract,
            ROUND(AVG(TotalCharges), 2) AS AvgContractRevenue
        FROM customers
        GROUP BY Contract
    ),
    OverallAvg AS (
        SELECT ROUND(AVG(TotalCharges), 2) AS OverallAvgRevenue
        FROM customers
    )
    SELECT 
        c.Contract,
        c.AvgContractRevenue,
        o.OverallAvgRevenue,
        ROUND(c.AvgContractRevenue - o.OverallAvgRevenue, 2) AS DifferenceFromOverall
    FROM ContractAvg c, OverallAvg o
    ORDER BY c.AvgContractRevenue DESC;
    """
    st.code(q9_sql, language="sql")
    st.dataframe(run_sql(q9_sql), width="stretch")
    st.info("💡 **Insight:** Month-to-month contracts bring in far less total lifetime revenue compared to the overall average, whereas one and two-year contracts exceed the average substantially.")

    # 7. Window Function (Top 20 Ranked)
    st.subheader("7. Window Function")
    q10_sql = """
    SELECT 
        customerID,
        TotalCharges,
        RANK() OVER(ORDER BY TotalCharges DESC) AS Rank,
        ROW_NUMBER() OVER(ORDER BY TotalCharges DESC) AS RowNumber
    FROM customers
    WHERE TotalCharges IS NOT NULL
    LIMIT 20;
    """
    st.code(q10_sql, language="sql")
    st.dataframe(run_sql(q10_sql), width="stretch")
    st.info("💡 **Insight:** Utilizing RANK and ROW_NUMBER allows us to assign proper leaderboard positions for the absolute highest revenue drivers.")

    # 8. DENSE_RANK (Payment Methods by Churn)
    st.subheader("8. DENSE_RANK")
    q11_sql = """
    WITH PaymentChurn AS (
        SELECT 
            PaymentMethod,
            ROUND(AVG(Churn_binary) * 100, 2) AS ChurnRate
        FROM customers
        GROUP BY PaymentMethod
    )
    SELECT 
        PaymentMethod,
        ChurnRate,
        DENSE_RANK() OVER(ORDER BY ChurnRate DESC) AS Rank
    FROM PaymentChurn;
    """
    st.code(q11_sql, language="sql")
    st.dataframe(run_sql(q11_sql), width="stretch")
    st.info("💡 **Insight:** DENSE_RANK clearly establishes 'Electronic check' as the definitive highest-churn category (Rank 1).")

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
# TAB 5 — Churn Model (Enhanced Evaluation)
# =====================================================================
with tab_ml:
    st.header("Churn Prediction Models")

    # ── Prepare features (unchanged) ───────────────────────────────────
    drop_cols = ["customerID", "Churn", "Churn_binary"]
    feature_df = df.drop(columns=drop_cols)
    feature_df = pd.get_dummies(feature_df, drop_first=True)
    X = feature_df.values
    y = df["Churn_binary"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ── Logistic Regression (scaled) — unchanged ───────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_preds = lr.predict(X_test_sc)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]

    # ── Random Forest — unchanged ──────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    # ── 1. Metrics Comparison Table ────────────────────────────────────
    st.subheader("📋 Model Comparison Table")

    def compute_metrics(y_true, y_pred, y_prob):
        """Compute all classification metrics for a model."""
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "ROC-AUC": roc_auc_score(y_true, y_prob),
        }

    lr_metrics = compute_metrics(y_test, lr_preds, lr_proba)
    rf_metrics = compute_metrics(y_test, rf_preds, rf_proba)

    metrics_df = pd.DataFrame(
        {"Logistic Regression": lr_metrics, "Random Forest": rf_metrics}
    ).T
    # Format as percentages for display
    metrics_display = metrics_df.style.format("{:.4f}").highlight_max(
        axis=0, props="color: white; background-color: #2ecc71; font-weight: bold;"
    )
    st.dataframe(metrics_display, width="stretch")

    # Metric cards row
    st.markdown("**Best Model per Metric:**")
    metric_cols = st.columns(5)
    for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]):
        best = "LR" if lr_metrics[metric] >= rf_metrics[metric] else "RF"
        best_val = max(lr_metrics[metric], rf_metrics[metric])
        metric_cols[i].metric(metric, f"{best_val:.4f}", delta=best)

    st.divider()

    # ── 2. Confusion Matrices ──────────────────────────────────────────
    st.subheader("🔢 Confusion Matrices")

    cm_col1, cm_col2 = st.columns(2)

    # Logistic Regression confusion matrix
    lr_cm = confusion_matrix(y_test, lr_preds)
    with cm_col1:
        st.markdown("**Logistic Regression**")
        fig_lr_cm = px.imshow(
            lr_cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Not Churned", "Churned"],
            y=["Not Churned", "Churned"],
            color_continuous_scale="Blues",
            aspect="equal",
        )
        fig_lr_cm.update_layout(height=350, margin=dict(t=30, b=30))
        st.plotly_chart(fig_lr_cm, width="stretch")

    # Random Forest confusion matrix
    rf_cm = confusion_matrix(y_test, rf_preds)
    with cm_col2:
        st.markdown("**Random Forest**")
        fig_rf_cm = px.imshow(
            rf_cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Not Churned", "Churned"],
            y=["Not Churned", "Churned"],
            color_continuous_scale="Oranges",
            aspect="equal",
        )
        fig_rf_cm.update_layout(height=350, margin=dict(t=30, b=30))
        st.plotly_chart(fig_rf_cm, width="stretch")

    st.divider()

    # ── 3. ROC Curves ──────────────────────────────────────────────────
    st.subheader("📈 ROC Curves")

    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=lr_fpr, y=lr_tpr, mode="lines",
        name=f"Logistic Regression (AUC = {lr_metrics['ROC-AUC']:.4f})",
        line=dict(color="#636EFA", width=2),
    ))
    fig_roc.add_trace(go.Scatter(
        x=rf_fpr, y=rf_tpr, mode="lines",
        name=f"Random Forest (AUC = {rf_metrics['ROC-AUC']:.4f})",
        line=dict(color="#EF553B", width=2),
    ))
    # Diagonal reference line
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Classifier",
        line=dict(color="gray", width=1, dash="dash"),
    ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.4, y=0.05),
        height=450,
    )
    st.plotly_chart(fig_roc, width="stretch")

    st.divider()

    # ── 4. Classification Reports ──────────────────────────────────────
    st.subheader("📄 Classification Reports")

    report_col1, report_col2 = st.columns(2)
    with report_col1:
        st.markdown("**Logistic Regression**")
        lr_report = classification_report(
            y_test, lr_preds, target_names=["Not Churned", "Churned"],
            output_dict=True,
        )
        st.dataframe(
            pd.DataFrame(lr_report).T.style.format(
                "{:.4f}", subset=pd.IndexSlice[:, ["precision", "recall", "f1-score"]]
            ),
            width="stretch",
        )

    with report_col2:
        st.markdown("**Random Forest**")
        rf_report = classification_report(
            y_test, rf_preds, target_names=["Not Churned", "Churned"],
            output_dict=True,
        )
        st.dataframe(
            pd.DataFrame(rf_report).T.style.format(
                "{:.4f}", subset=pd.IndexSlice[:, ["precision", "recall", "f1-score"]]
            ),
            width="stretch",
        )

    st.divider()

    # ── 5. Feature Importances (unchanged) ─────────────────────────────
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


# =====================================================================
# TAB 6 — Predict Customer Churn
# =====================================================================
with tab_predict:
    st.header("🔮 Predict Customer Churn")
    st.markdown(
        "Enter customer information below to predict whether the customer "
        "is likely to churn."
    )

    # ── Determine best model based on ROC-AUC from tab_ml ──────────────
    if lr_metrics["ROC-AUC"] >= rf_metrics["ROC-AUC"]:
        best_model_name = "Logistic Regression"
        best_model = lr
        use_scaling = True
    else:
        best_model_name = "Random Forest"
        best_model = rf
        use_scaling = False

    st.info(f"Using **{best_model_name}** (best ROC-AUC) for predictions.")

    # ── Input Form ─────────────────────────────────────────────────────
    with st.form("churn_prediction_form"):

        # --- Demographics -----------------------------------------------
        st.subheader("👤 Demographics")
        demo_col1, demo_col2, demo_col3, demo_col4 = st.columns(4)
        with demo_col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with demo_col2:
            senior_citizen = st.radio("Senior Citizen", ["No", "Yes"])
        with demo_col3:
            partner = st.selectbox("Partner", ["No", "Yes"])
        with demo_col4:
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        st.divider()

        # --- Account Information ----------------------------------------
        st.subheader("📋 Account Information")
        acct_col1, acct_col2 = st.columns(2)
        with acct_col1:
            tenure = st.slider("Tenure (months)", min_value=1, max_value=72, value=12)
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
        with acct_col2:
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                ],
            )

        st.divider()

        # --- Services ---------------------------------------------------
        st.subheader("📞 Phone & Internet Services")
        svc_col1, svc_col2, svc_col3 = st.columns(3)
        with svc_col1:
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )
        with svc_col2:
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"]
            )
        with svc_col3:
            online_backup = st.selectbox(
                "Online Backup", ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"]
            )

        st.subheader("🎬 Streaming & Support")
        str_col1, str_col2, str_col3 = st.columns(3)
        with str_col1:
            tech_support = st.selectbox(
                "Tech Support", ["No", "Yes", "No internet service"]
            )
        with str_col2:
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
        with str_col3:
            streaming_movies = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"]
            )

        st.divider()

        # --- Charges ----------------------------------------------------
        st.subheader("💰 Charges")
        charge_col1, charge_col2 = st.columns(2)
        with charge_col1:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=18.0, max_value=120.0, value=50.0, step=0.5,
            )
        with charge_col2:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0, max_value=9000.0, value=600.0, step=10.0,
            )

        submitted = st.form_submit_button(
            "🔍 Predict Churn", type="primary", use_container_width=True
        )

    # ── Run prediction when form is submitted ──────────────────────────
    if submitted:
        # Build a single-row DataFrame matching training columns
        input_dict = {
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
        }
        input_df = pd.DataFrame([input_dict])

        # Apply the same get_dummies encoding as training
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Align columns to match training feature set (add missing, drop extra)
        input_encoded = input_encoded.reindex(
            columns=feature_df.columns, fill_value=0
        )

        X_input = input_encoded.values

        # Scale if the best model requires it (Logistic Regression)
        if use_scaling:
            X_input = scaler.transform(X_input)

        # Predict
        prediction = best_model.predict(X_input)[0]
        proba = best_model.predict_proba(X_input)[0]
        churn_prob = proba[1]
        stay_prob = proba[0]

        st.divider()

        # ── Display Results ────────────────────────────────────────────
        st.subheader("📊 Prediction Results")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("#### Prediction")
            if prediction == 1:
                st.error("⚠️ **Likely to Churn**")
            else:
                st.success("✅ **Likely to Stay**")

        with res_col2:
            st.markdown("#### Churn Probability")
            st.metric("Churn Probability", f"{churn_prob:.1%}")
            st.progress(min(churn_prob, 1.0))

        st.divider()

        # ── Risk Classification ────────────────────────────────────────
        st.subheader("🎯 Risk Classification")

        if churn_prob <= 0.30:
            risk_level = "Low"
            st.success(f"🟢 **Low Risk** — Churn probability: {churn_prob:.1%}")
        elif churn_prob <= 0.70:
            risk_level = "Medium"
            st.warning(f"🟡 **Medium Risk** — Churn probability: {churn_prob:.1%}")
        else:
            risk_level = "High"
            st.error(f"🔴 **High Risk** — Churn probability: {churn_prob:.1%}")

        st.divider()

        # ── Business Recommendations ───────────────────────────────────
        st.subheader("💡 Business Recommendations")

        if risk_level == "High":
            recommendation_text = "Offer annual contract, provide loyalty discount, assign retention specialist, contact within 48 hours, bundle services."
            st.error("🔴 **High Risk — Immediate Action Required**")
            st.markdown(
                """
                - 📝 **Offer an annual contract** with a discounted rate to lock in commitment
                - 💰 **Provide a loyalty discount** (10–20% off) to incentivize staying
                - 👤 **Assign a dedicated retention specialist** for personalized outreach
                - ⏰ **Contact the customer within 48 hours** before they finalize their decision
                - 🎁 **Bundle complementary services** (e.g., free tech support for 3 months)
                """
            )
        elif risk_level == "Medium":
            recommendation_text = "Offer promotional bundle, send personalized email, monitor engagement, suggest contract upgrade, schedule check-in call."
            st.warning("🟡 **Medium Risk — Proactive Engagement Recommended**")
            st.markdown(
                """
                - 📦 **Offer a promotional bundle** with added value (streaming, security add-ons)
                - 📧 **Send a personalized email** highlighting their usage benefits and savings
                - 📊 **Monitor engagement metrics** closely over the next billing cycle
                - 🔄 **Suggest a contract upgrade** with a modest discount for longer commitment
                - 💬 **Schedule a check-in call** to address any service concerns
                """
            )
        else:
            recommendation_text = "Maintain current relationship, recommend premium services, encourage referrals, continue current plan, upsell opportunities."
            st.success("🟢 **Low Risk — Maintain & Grow**")
            st.markdown(
                """
                - 🤝 **Maintain the current relationship** — the customer is satisfied
                - ⭐ **Recommend premium services** (higher-tier plans, add-ons)
                - 📣 **Encourage referrals** with a referral bonus program
                - ✅ **Continue the current plan** — no immediate intervention needed
                - 🎯 **Upsell opportunities** — suggest relevant upgrades based on usage
                """
            )

        st.divider()

        # ── Download Prediction Report ─────────────────────────────────
        st.subheader("📥 Download Report")
        
        report_data = {
            "Gender": gender,
            "Senior Citizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure (months)": tenure,
            "Contract": contract,
            "Paperless Billing": paperless_billing,
            "Payment Method": payment_method,
            "Phone Service": phone_service,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet_service,
            "Online Security": online_security,
            "Online Backup": online_backup,
            "Device Protection": device_protection,
            "Tech Support": tech_support,
            "Streaming TV": streaming_tv,
            "Streaming Movies": streaming_movies,
            "Monthly Charges ($)": monthly_charges,
            "Total Charges ($)": total_charges,
            "Prediction": "Likely to Churn" if prediction == 1 else "Likely to Stay",
            "Probability": f"{churn_prob:.2%}",
            "Risk Level": risk_level,
            "Recommendation": recommendation_text,
        }
        
        report_df = pd.DataFrame(list(report_data.items()), columns=["Field", "Value"])
        csv_data = report_df.to_csv(index=False)
        
        filename = f"customer_churn_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        st.download_button(
            label="📄 Download Prediction Report",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            use_container_width=True
        )
