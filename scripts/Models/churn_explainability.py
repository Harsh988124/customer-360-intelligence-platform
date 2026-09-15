"""
Explainable AI - SHAP
Customer 360 Intelligence & Retention Platform

Explains the Logistic Regression churn model.

Outputs:
1. Global feature importance
2. Customer-level SHAP explanations
3. Top churn drivers for each customer

Important:
- churn_probability is NOT used
- customer_status is NOT used
"""

import os

import numpy as np
import pandas as pd
import shap

from dotenv import load_dotenv
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------

def get_database_url():

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    if not all([host, port, database, username, password]):
        raise ValueError(
            "Missing database environment variables."
        )

    return (
        f"postgresql+psycopg2://{username}:{password}"
        f"@{host}:{port}/{database}"
    )


# ---------------------------------------------------------
# LOAD CUSTOMER DATA
# ---------------------------------------------------------

def load_customer_data(engine):

    query = """
        SELECT
            customer_id,
            age,
            monthly_price,
            subscription_count,
            auto_renew_count,
            total_transactions,
            successful_transactions,
            failed_transactions,
            total_spend,
            average_spend_per_transaction,
            transaction_success_rate,
            payment_failure_rate,
            usage_records,
            active_days,
            total_logins,
            total_session_minutes,
            total_features_used,
            average_features_used,
            average_session_minutes,
            total_support_tickets,
            average_resolution_hours,
            negative_tickets,
            negative_ticket_rate,
            tenure_days,
            tenure_months,
            support_burden,
            churned
        FROM customer_360
    """

    df = pd.read_sql(query, engine)

    print(f"Loaded {len(df):,} customers.")

    return df


# ---------------------------------------------------------
# FEATURE LIST
# ---------------------------------------------------------

FEATURES = [
    "age",
    "monthly_price",
    "subscription_count",
    "auto_renew_count",
    "total_transactions",
    "successful_transactions",
    "failed_transactions",
    "total_spend",
    "average_spend_per_transaction",
    "transaction_success_rate",
    "payment_failure_rate",
    "usage_records",
    "active_days",
    "total_logins",
    "total_session_minutes",
    "total_features_used",
    "average_features_used",
    "average_session_minutes",
    "total_support_tickets",
    "average_resolution_hours",
    "negative_tickets",
    "negative_ticket_rate",
    "tenure_days",
    "tenure_months",
    "support_burden",
]


# ---------------------------------------------------------
# CREATE LOGISTIC REGRESSION MODEL
# ---------------------------------------------------------

def create_model():

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    return model


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

def train_model(df):

    X = df[FEATURES].copy()
    y = df["churned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = create_model()

    model.fit(X_train, y_train)

    print("\nLogistic Regression model trained successfully.")

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")

    return model, X, y


# ---------------------------------------------------------
# PREPARE DATA FOR SHAP
# ---------------------------------------------------------

def prepare_shap_data(model, X):

    # Get preprocessing pipeline
    preprocessing = model[:-1]

    # Transform the complete feature dataset
    X_transformed = preprocessing.transform(X)

    # Get final logistic regression model
    classifier = model.named_steps["model"]

    return X_transformed, classifier


# ---------------------------------------------------------
# CALCULATE SHAP VALUES
# ---------------------------------------------------------

def calculate_shap_values(model, X):

    X_transformed, classifier = prepare_shap_data(
        model,
        X
    )

    print("\nCalculating SHAP values...")

    explainer = shap.LinearExplainer(
        classifier,
        X_transformed
    )

    shap_values = explainer.shap_values(
        X_transformed
    )

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)

    print("SHAP values calculated successfully.")

    return shap_values


# ---------------------------------------------------------
# GLOBAL FEATURE IMPORTANCE
# ---------------------------------------------------------

def create_global_importance(
    shap_values,
    engine
):

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "mean_absolute_shap": mean_abs_shap
    })

    importance_df = importance_df.sort_values(
        "mean_absolute_shap",
        ascending=False
    ).reset_index(drop=True)

    importance_df["importance_rank"] = (
        importance_df.index + 1
    )

    importance_df["mean_absolute_shap"] = (
        importance_df["mean_absolute_shap"].round(6)
    )

    importance_df.to_sql(
        "shap_feature_importance",
        engine,
        if_exists="replace",
        index=False,
        method="multi"
    )

    print(
        "\nGlobal SHAP feature importance "
        "saved to PostgreSQL."
    )

    return importance_df


# ---------------------------------------------------------
# CUSTOMER-LEVEL EXPLANATIONS
# ---------------------------------------------------------

def create_customer_explanations(
    df,
    shap_values,
    engine
):

    records = []

    for row_index in range(len(df)):

        customer_id = df.iloc[row_index]["customer_id"]

        customer_shap = shap_values[row_index]

        # Find top 5 absolute SHAP drivers
        top_indices = np.argsort(
            np.abs(customer_shap)
        )[::-1][:5]

        for rank, feature_index in enumerate(
            top_indices,
            start=1
        ):

            shap_value = customer_shap[feature_index]

            direction = (
                "Increases Churn Risk"
                if shap_value > 0
                else "Decreases Churn Risk"
            )

            records.append({
                "customer_id": customer_id,
                "feature": FEATURES[feature_index],
                "shap_value": round(
                    float(shap_value),
                    6
                ),
                "impact_direction": direction,
                "importance_rank": rank
            })

    explanations_df = pd.DataFrame(records)

    explanations_df.to_sql(
        "customer_shap_explanations",
        engine,
        if_exists="replace",
        index=False,
        method="multi"
    )

    print(
        "\nCustomer-level SHAP explanations "
        "saved to PostgreSQL."
    )

    return explanations_df


# ---------------------------------------------------------
# PRINT TOP FEATURES
# ---------------------------------------------------------

def print_global_summary(importance_df):

    print("\n" + "=" * 60)
    print("TOP CHURN DRIVERS")
    print("=" * 60)

    print(
        importance_df[
            [
                "importance_rank",
                "feature",
                "mean_absolute_shap"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


# ---------------------------------------------------------
# PRINT SAMPLE CUSTOMER EXPLANATION
# ---------------------------------------------------------

def print_sample_explanation(
    df,
    shap_values
):

    # Select customer with highest predicted churn
    model = create_model()

    X = df[FEATURES].copy()
    y = df["churned"].astype(int)

    model.fit(X, y)

    probabilities = model.predict_proba(X)[:, 1]

    highest_risk_index = np.argmax(
        probabilities
    )

    customer_id = df.iloc[
        highest_risk_index
    ]["customer_id"]

    customer_shap = shap_values[
        highest_risk_index
    ]

    top_indices = np.argsort(
        np.abs(customer_shap)
    )[::-1][:5]

    print("\n" + "=" * 60)
    print("SAMPLE CUSTOMER EXPLANATION")
    print("=" * 60)

    print(f"Customer ID: {customer_id}")
    print(
        f"Predicted Churn Probability: "
        f"{probabilities[highest_risk_index]:.4f}"
    )

    print("\nTop factors:")

    for rank, feature_index in enumerate(
        top_indices,
        start=1
    ):

        value = customer_shap[feature_index]

        direction = (
            "Increases Churn Risk"
            if value > 0
            else "Decreases Churn Risk"
        )

        print(
            f"{rank}. "
            f"{FEATURES[feature_index]} "
            f"({value:.4f}) → "
            f"{direction}"
        )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("SHAP CHURN EXPLAINABILITY PIPELINE")
    print("=" * 60)

    database_url = get_database_url()

    engine = create_engine(
        database_url
    )

    # Load data
    print("\nLoading Customer 360 data...")

    df = load_customer_data(
        engine
    )

    # Train model
    model, X, y = train_model(
        df
    )

    # SHAP
    shap_values = calculate_shap_values(
        model,
        X
    )

    # Global importance
    importance_df = create_global_importance(
        shap_values,
        engine
    )

    # Customer explanations
    create_customer_explanations(
        df,
        shap_values,
        engine
    )

    # Print results
    print_global_summary(
        importance_df
    )

    print_sample_explanation(
        df,
        shap_values
    )

    print("\n" + "=" * 60)
    print("PHASE 10 SHAP EXPLAINABILITY COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()