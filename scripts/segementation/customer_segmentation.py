"""
Customer Segmentation using K-Means
Customer 360 Intelligence & Retention Platform

Phase 7:
- Load Customer 360 data from PostgreSQL
- Prepare behavioral and value features
- Standardize features
- Evaluate K-Means for K=2 to K=8
- Select the best K using Silhouette Score
- Train final K-Means model
- Profile customer clusters
- Assign business-friendly segment names
- Save results to PostgreSQL
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------

load_dotenv()


def get_database_config():
    """
    Read PostgreSQL configuration from .env
    """

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    required = {
        "DB_HOST": host,
        "DB_NAME": database,
        "DB_USER": username,
        "DB_PASSWORD": password,
    }

    missing = [
        key for key, value in required.items()
        if not value
    ]

    if missing:
        raise ValueError(
            f"Missing database environment variables: {missing}"
        )

    return {
        "host": host,
        "port": port,
        "database": database,
        "username": username,
        "password": password,
    }


def create_db_engine():
    """
    Create SQLAlchemy PostgreSQL connection.
    """

    config = get_database_config()

    connection_string = (
        f"postgresql+psycopg2://"
        f"{config['username']}:{config['password']}"
        f"@{config['host']}:{config['port']}"
        f"/{config['database']}"
    )

    return create_engine(connection_string)


# ---------------------------------------------------------
# LOAD CUSTOMER 360
# ---------------------------------------------------------

def load_customer_360():
    """
    Load the curated Customer 360 table from PostgreSQL.
    """

    print("\n" + "=" * 70)
    print("LOADING CUSTOMER 360 DATA")
    print("=" * 70)

    engine = create_db_engine()

    query = """
        SELECT *
        FROM customer_360
        ORDER BY customer_id
    """

    df = pd.read_sql(query, engine)

    print(f"Customer 360 rows loaded: {len(df):,}")
    print(f"Customer 360 columns: {len(df.columns)}")

    return df


# ---------------------------------------------------------
# FEATURE PREPARATION
# ---------------------------------------------------------

def prepare_features(df):
    """
    Select features appropriate for customer segmentation.

    Important:
    Do NOT use churned, churn_probability, or customer_status.
    """

    print("\n" + "=" * 70)
    print("PREPARING SEGMENTATION FEATURES")
    print("=" * 70)

    feature_columns = [
        "total_spend",
        "total_transactions",
        "total_logins",
        "total_session_minutes",
        "average_features_used",
        "total_support_tickets",
        "tenure_months",
        "payment_failure_rate",
        "negative_ticket_rate",
    ]

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required features are missing "
            f"from customer_360: {missing_columns}"
        )

    features = df[feature_columns].copy()

    # Convert values to numeric.
    for column in feature_columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce"
        )

    # Replace missing and infinite values.
    features = features.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    features = features.fillna(0)

    print("\nFeatures used for K-Means:")

    for column in feature_columns:
        print(f"  - {column}")

    print("\nFeature matrix shape:")
    print(features.shape)

    return features, feature_columns


# ---------------------------------------------------------
# SCALE FEATURES
# ---------------------------------------------------------

def scale_features(features):
    """
    Standardize features so large-value variables
    do not dominate K-Means.
    """

    print("\n" + "=" * 70)
    print("STANDARDIZING FEATURES")
    print("=" * 70)

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(features)

    print("StandardScaler applied successfully.")

    return scaled_features, scaler


# ---------------------------------------------------------
# FIND BEST K
# ---------------------------------------------------------

def find_best_k(scaled_features):
    """
    Test K values from 2 through 8.

    Selection metric:
    Silhouette Score

    Higher score generally means better-separated clusters.
    """

    print("\n" + "=" * 70)
    print("EVALUATING K-MEANS CLUSTERS")
    print("=" * 70)

    results = []

    for k in range(2, 9):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(scaled_features)

        silhouette = silhouette_score(
            scaled_features,
            labels
        )

        inertia = model.inertia_

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": silhouette
        })

        print(
            f"K={k} | "
            f"Inertia={inertia:,.2f} | "
            f"Silhouette={silhouette:.4f}"
        )

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["silhouette_score"].idxmax()
    ]

    best_k = int(best_row["k"])

    print("\n" + "-" * 70)
    print(f"BEST K: {best_k}")
    print(
        f"BEST SILHOUETTE SCORE: "
        f"{best_row['silhouette_score']:.4f}"
    )
    print("-" * 70)

    return best_k, results_df


# ---------------------------------------------------------
# TRAIN FINAL MODEL
# ---------------------------------------------------------

def train_final_model(scaled_features, best_k):
    """
    Train final K-Means model using the selected K.
    """

    print("\n" + "=" * 70)
    print("TRAINING FINAL K-MEANS MODEL")
    print("=" * 70)

    model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(scaled_features)

    print("Final K-Means model trained successfully.")
    print(f"Number of clusters: {best_k}")

    return model, labels


# ---------------------------------------------------------
# BUILD SEGMENT DATASET
# ---------------------------------------------------------

def build_segment_dataset(
    df,
    labels,
    feature_columns
):
    """
    Add cluster assignments to the customer dataset.
    """

    print("\n" + "=" * 70)
    print("CREATING CUSTOMER SEGMENTS")
    print("=" * 70)

    result = df.copy()

    result["cluster_id"] = labels

    print("\nCustomers per cluster:")

    cluster_counts = (
        result["cluster_id"]
        .value_counts()
        .sort_index()
    )

    for cluster_id, count in cluster_counts.items():
        print(
            f"  Cluster {cluster_id}: "
            f"{count:,} customers"
        )

    return result


# ---------------------------------------------------------
# CLUSTER PROFILING
# ---------------------------------------------------------

def profile_clusters(
    segment_df,
    feature_columns
):
    """
    Calculate average feature values for every cluster.
    """

    print("\n" + "=" * 70)
    print("PROFILING CUSTOMER CLUSTERS")
    print("=" * 70)

    profile = (
        segment_df
        .groupby("cluster_id")[feature_columns]
        .mean()
        .round(2)
    )

    print("\nCluster profile:")
    print(profile)

    return profile


# ---------------------------------------------------------
# BUSINESS SEGMENT LABELS
# ---------------------------------------------------------

def assign_segment_names(
    segment_df,
    profile
):
    """
    Convert technical cluster IDs into
    business-friendly customer segments.

    The naming is based on relative cluster behavior,
    not hard-coded customer rules.
    """

    print("\n" + "=" * 70)
    print("ASSIGNING BUSINESS SEGMENT NAMES")
    print("=" * 70)

    profile = profile.copy()

    # Normalize selected profile metrics to help
    # compare clusters.
    profile["spend_rank"] = (
        profile["total_spend"].rank(
            method="first",
            ascending=False
        )
    )

    profile["engagement_score"] = (
        profile["total_logins"]
        + profile["total_session_minutes"]
        + profile["average_features_used"]
    )

    profile["engagement_rank"] = (
        profile["engagement_score"].rank(
            method="first",
            ascending=False
        )
    )

    profile["support_rank"] = (
        profile["total_support_tickets"].rank(
            method="first",
            ascending=False
        )
    )

    profile["tenure_rank"] = (
        profile["tenure_months"].rank(
            method="first",
            ascending=True
        )
    )

    # Identify the strongest value/engagement cluster.
    premium_cluster = (
        profile["total_spend"]
        + profile["engagement_score"]
    ).idxmax()

    remaining = [
        cluster
        for cluster in profile.index
        if cluster != premium_cluster
    ]

    # Highest support burden among remaining clusters.
    if remaining:
        risk_cluster = (
            profile.loc[remaining, "total_support_tickets"]
            + profile.loc[remaining, "payment_failure_rate"]
            + profile.loc[remaining, "negative_ticket_rate"]
        ).idxmax()
    else:
        risk_cluster = premium_cluster

    remaining = [
        cluster
        for cluster in remaining
        if cluster != risk_cluster
    ]

    # Lowest engagement among remaining clusters.
    if remaining:
        low_engagement_cluster = (
            profile.loc[remaining, "engagement_score"]
        ).idxmin()
    else:
        low_engagement_cluster = None

    segment_mapping = {}

    segment_mapping[premium_cluster] = "Premium Loyal"

    if risk_cluster != premium_cluster:
        segment_mapping[risk_cluster] = "High Value At Risk"

    if low_engagement_cluster is not None:
        segment_mapping[
            low_engagement_cluster
        ] = "Low Engagement"

    # Any remaining clusters become Regular Customers.
    for cluster in profile.index:
        if cluster not in segment_mapping:
            segment_mapping[cluster] = "Regular Customers"

    segment_df["segment_name"] = (
        segment_df["cluster_id"]
        .map(segment_mapping)
    )

    print("\nCluster → Segment mapping:")

    for cluster_id, segment_name in sorted(
        segment_mapping.items()
    ):
        print(
            f"  Cluster {cluster_id} → "
            f"{segment_name}"
        )

    return segment_df, segment_mapping


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

def save_results(
    segment_df,
    results_df,
    profile,
    engine
):
    """
    Save segmentation outputs to PostgreSQL.
    """

    print("\n" + "=" * 70)
    print("SAVING SEGMENTATION RESULTS")
    print("=" * 70)

    # Customer-level segmentation.
    customer_output = segment_df[
        [
            "customer_id",
            "cluster_id",
            "segment_name"
        ]
    ].copy()

    customer_output.to_sql(
        "customer_segments",
        engine,
        if_exists="replace",
        index=False,
        method="multi"
    )

    # K evaluation results.
    results_df.to_sql(
        "kmeans_evaluation",
        engine,
        if_exists="replace",
        index=False,
        method="multi"
    )

    # Cluster profiles.
    profile_output = profile.reset_index()

    profile_output.to_sql(
        "segment_profiles",
        engine,
        if_exists="replace",
        index=False,
        method="multi"
    )

    print("Saved PostgreSQL tables:")
    print("  - customer_segments")
    print("  - kmeans_evaluation")
    print("  - segment_profiles")


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

def show_final_summary(
    segment_df,
    best_k,
    results_df
):
    """
    Display final segmentation summary.
    """

    print("\n" + "=" * 70)
    print("FINAL SEGMENTATION SUMMARY")
    print("=" * 70)

    print(f"Total customers: {len(segment_df):,}")
    print(f"Selected K: {best_k}")

    best_score = results_df.loc[
        results_df["k"] == best_k,
        "silhouette_score"
    ].iloc[0]

    print(
        f"Silhouette Score: {best_score:.4f}"
    )

    print("\nSegment distribution:")

    distribution = (
        segment_df["segment_name"]
        .value_counts()
    )

    for segment, count in distribution.items():

        percentage = (
            count / len(segment_df)
        ) * 100

        print(
            f"  {segment}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main():

    print("\n")
    print("=" * 70)
    print("PHASE 7 - CUSTOMER SEGMENTATION")
    print("K-MEANS CLUSTERING")
    print("=" * 70)

    try:

        # 1. Load Customer 360
        df = load_customer_360()

        # 2. Prepare features
        features, feature_columns = prepare_features(df)

        # 3. Scale features
        scaled_features, scaler = scale_features(
            features
        )

        # 4. Find optimal K
        best_k, results_df = find_best_k(
            scaled_features
        )

        # 5. Train final model
        model, labels = train_final_model(
            scaled_features,
            best_k
        )

        # 6. Build segmentation dataset
        segment_df = build_segment_dataset(
            df,
            labels,
            feature_columns
        )

        # 7. Profile clusters
        profile = profile_clusters(
            segment_df,
            feature_columns
        )

        # 8. Assign business names
        segment_df, mapping = assign_segment_names(
            segment_df,
            profile
        )

        # 9. Database connection
        engine = create_db_engine()

        # 10. Save results
        save_results(
            segment_df,
            results_df,
            profile,
            engine
        )

        # 11. Final summary
        show_final_summary(
            segment_df,
            best_k,
            results_df
        )

        print("\n" + "=" * 70)
        print("PHASE 7 COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as error:

        print("\n" + "=" * 70)
        print("PHASE 7 FAILED")
        print("=" * 70)

        print(f"Error: {error}")

        raise


if __name__ == "__main__":
    main()