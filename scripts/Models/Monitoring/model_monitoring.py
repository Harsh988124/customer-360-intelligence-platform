import os
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from dotenv import load_dotenv


# ---------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "customer_intelligence")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "customer360pass")


# ---------------------------------------------------------
# 2. Connect to Docker PostgreSQL
# ---------------------------------------------------------

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ---------------------------------------------------------
# 3. Load churn predictions
# ---------------------------------------------------------

query = """
SELECT
    customer_id,
    churn_probability_ml,
    predicted_churn
FROM churn_predictions
"""

df = pd.read_sql(query, engine)

print("\n========================================")
print("MODEL MONITORING")
print("========================================")

print(f"Total predictions: {len(df):,}")


# ---------------------------------------------------------
# 4. Load actual churn labels
# ---------------------------------------------------------

actual_query = """
SELECT
    customer_id,
    churned
FROM customer_360
"""

actual_df = pd.read_sql(actual_query, engine)

df = df.merge(
    actual_df,
    on="customer_id",
    how="inner"
)

print(f"Predictions matched with actual labels: {len(df):,}")


# ---------------------------------------------------------
# 5. Calculate monitoring metrics
# ---------------------------------------------------------

y_true = df["churned"]
y_pred = df["predicted_churn"]
y_prob = df["churn_probability_ml"]


accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)


prediction_count = len(df)

predicted_churn_count = int(y_pred.sum())

predicted_churn_rate = (
    predicted_churn_count / prediction_count
)

actual_churn_rate = y_true.mean()

average_probability = y_prob.mean()

minimum_probability = y_prob.min()

maximum_probability = y_prob.max()


# ---------------------------------------------------------
# 6. Display results
# ---------------------------------------------------------

print("\nModel: Logistic Regression")
print("Model Version: v1")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print(f"\nAverage churn probability: {average_probability:.4f}")
print(f"Minimum churn probability: {minimum_probability:.4f}")
print(f"Maximum churn probability: {maximum_probability:.4f}")

print(f"\nPredicted churn customers: {predicted_churn_count:,}")
print(f"Predicted churn rate: {predicted_churn_rate:.2%}")
print(f"Actual churn rate: {actual_churn_rate:.2%}")


# ---------------------------------------------------------
# 7. Create monitoring record
# ---------------------------------------------------------

monitoring_record = pd.DataFrame([
    {
        "monitoring_time": datetime.now(),

        "model_name": "Logistic Regression",

        "model_version": "v1",

        "prediction_count": prediction_count,

        "average_churn_probability": average_probability,

        "minimum_churn_probability": minimum_probability,

        "maximum_churn_probability": maximum_probability,

        "predicted_churn_count": predicted_churn_count,

        "predicted_churn_rate": predicted_churn_rate,

        "actual_churn_rate": actual_churn_rate,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "roc_auc": roc_auc,
    }
])


# ---------------------------------------------------------
# 8. Save monitoring results
# ---------------------------------------------------------

monitoring_record.to_sql(
    "model_monitoring",
    engine,
    if_exists="append",
    index=False
)


print("\n========================================")
print("Monitoring record saved successfully")
print("Table: model_monitoring")
print("========================================")