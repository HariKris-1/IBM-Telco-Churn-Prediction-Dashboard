import sqlite3
import pandas as pd

def test(q, name):
    print(f"--- {name} ---")
    try:
        conn = sqlite3.connect('churn.db')
        df = pd.read_sql(q, conn)
        print(df.head())
        print("Success!\n")
    except Exception as e:
        print("Error:", e, "\n")
    finally:
        conn.close()

q1 = """
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
test(q1, "1. Customer Segmentation")

q2 = """
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
test(q2, "2. Revenue Analysis")

q3 = """
SELECT 
    PaymentMethod,
    COUNT(*) AS TotalCustomers,
    ROUND(AVG(Churn_binary) * 100, 2) AS ChurnRate
FROM customers
GROUP BY PaymentMethod
ORDER BY ChurnRate DESC;
"""
test(q3, "3. Churn by Payment Method")

q4 = """
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
test(q4, "4. Highest Revenue Customers")

q5 = """
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
test(q5, "5. CASE WHEN Analysis")

q6 = """
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
test(q6, "6. CTE Analysis")

q7 = """
SELECT 
    customerID,
    TotalCharges,
    RANK() OVER(ORDER BY TotalCharges DESC) AS Rank,
    ROW_NUMBER() OVER(ORDER BY TotalCharges DESC) AS RowNumber
FROM customers
WHERE TotalCharges IS NOT NULL
LIMIT 20;
"""
test(q7, "7. Window Function")

q8 = """
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
test(q8, "8. DENSE_RANK")

