"""
Churn Prediction
Customer 360 Intelligence & Retention Platform

Models:
1. Logistic Regression
2. Random Forest
3. XGBoost

Target:
- churned

Important:
- churn_probability is NOT used as a feature
- customer_status is NOT used as a feature
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


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
# LOAD DATA
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
# FEATURE SELECTION
# ---------------------------------------------------------

def prepare_data(df):

    features = [
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

    X = df[features].copy()
    y = df["churned"].astype(int)

    return X, y


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nData Split:")
    print(f"Training customers: {len(X_train):,}")
    print(f"Testing customers:  {len(X_test):,}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# CREATE MODELS
# ---------------------------------------------------------

def create_models(y_train):

    # Logistic Regression
    logistic_model = Pipeline(
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

    # Random Forest
    random_forest_model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_split=10,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    # Calculate class imbalance for XGBoost
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1
    )

    # XGBoost
    xgboost_model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    return {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
        "XGBoost": xgboost_model,
    }


# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------

def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        ),
    }

    return model, predictions, probabilities, metrics


# ---------------------------------------------------------
# SAVE MODEL RESULTS
# ---------------------------------------------------------

def save_predictions(
    df,
    X,
    model,
    model_name,
    engine
):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = model.predict(X)

    result = pd.DataFrame({
        "customer_id": df["customer_id"],
        "churn_probability_ml": probabilities.round(4),
        "predicted_churn": predictions
    })

    result["model_name"] = model_name

    table_name = "churn_predictions"

    result.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
        method="multi"
    )

    print(
        f"\nSaved {len(result):,} predictions "
        f"to {table_name}."
    )

    return result


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION PIPELINE")
    print("=" * 60)

    # Database
    database_url = get_database_url()
    engine = create_engine(database_url)

    # Load data
    print("\nLoading Customer 360 data...")
    df = load_customer_data(engine)

    # Target distribution
    print("\nChurn Distribution:")

    churn_distribution = (
        df["churned"]
        .value_counts()
        .sort_index()
    )

    print(churn_distribution)

    # Prepare features
    print("\nPreparing ML features...")

    X, y = prepare_data(df)

    print(f"Number of features: {X.shape[1]}")

    # Split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # Models
    models = create_models(y_train)

    results = []
    trained_models = {}

    # Train and evaluate
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        trained_model, predictions, probabilities, metrics = (
            evaluate_model(
                model,
                X_train,
                X_test,
                y_train,
                y_test
            )
        )

        trained_models[model_name] = trained_model

        results.append({
            "model": model_name,
            **metrics
        })

        print(f"{model_name} Results:")
        print(f"Accuracy : {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall   : {metrics['Recall']:.4f}")
        print(f"F1 Score : {metrics['F1 Score']:.4f}")
        print(f"ROC-AUC  : {metrics['ROC-AUC']:.4f}")

    # Model comparison
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Best model
    best_model_name = results_df.iloc[0]["model"]

    best_model = trained_models[best_model_name]

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(f"Best Model: {best_model_name}")

    print(
        f"Best ROC-AUC: "
        f"{results_df.iloc[0]['ROC-AUC']:.4f}"
    )

    # Save predictions from best model
    print("\nGenerating final churn predictions...")

    save_predictions(
        df,
        X,
        best_model,
        best_model_name,
        engine
    )

    # Save model evaluation
    results_df.to_sql(
        "churn_model_evaluation",
        engine,
        if_exists="replace",
        index=False,
        method="multi"
    )

    print("\nModel evaluation saved to PostgreSQL.")

    print("\n" + "=" * 60)
    print("PHASE 9 CHURN PREDICTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()