"""Load Telco Customer Churn CSV into SQLite database."""

import sqlite3
import pandas as pd

CSV_PATH = "data/telco_churn.csv"
DB_PATH = "churn.db"


def load_and_clean():
    df = pd.read_csv(CSV_PATH)

    # TotalCharges has whitespace strings for new customers — coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

    df["Churn_binary"] = (df["Churn"] == "Yes").astype(int)
    return df


def write_to_db(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    conn.close()


if __name__ == "__main__":
    df = load_and_clean()
    write_to_db(df)
    print(f"Loaded {len(df)} rows into {DB_PATH}")
