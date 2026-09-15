import pandas as pd
import numpy as np
import random
from faker import Faker
from pathlib import Path

# ==========================================
# 1. SETUP & INITIALIZATION
# ==========================================

fake = Faker("en_IN")

np.random.seed(42)
random.seed(42)
Faker.seed(42)

NUM_CUSTOMERS = 5000

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Starting synthetic data generation...")


# ==========================================
# 2. GENERATE CUSTOMERS
# ==========================================

print("Generating customers...")

customer_ids = [
    f"C{10000 + i}"
    for i in range(1, NUM_CUSTOMERS + 1)
]

cities = [
    "Mumbai",
    "Pune",
    "Nagpur",
    "Delhi",
    "Bengaluru",
    "Hyderabad",
    "Chennai",
    "Kolkata"
]

genders = [
    "Male",
    "Female",
    "Other"
]

customers = []

for cid in customer_ids:

    signup_date = fake.date_between(
        start_date="-3y",
        end_date="-1m"
    )

    customers.append({
        "customer_id": cid,
        "first_name": fake.first_name(),
        "last_name": fake.last_name(),
        "age": random.randint(18, 70),
        "gender": random.choice(genders),
        "city": random.choice(cities),
        "country": "India",
        "signup_date": signup_date,
        "customer_status": "Active"
    })


customers_df = pd.DataFrame(customers)


# ==========================================
# 3. GENERATE SUBSCRIPTIONS
# ==========================================

print("Generating subscriptions...")

plans = {
    "Basic": 499,
    "Standard": 999,
    "Premium": 1499
}

subscriptions = []

subscription_info = {}

for cid in customer_ids:

    plan = random.choice(
        list(plans.keys())
    )

    contract_type = random.choice(
        ["Monthly", "Yearly"]
    )

    auto_renew = random.choice(
        ["Yes", "No"]
    )

    subscription_info[cid] = {
        "plan_name": plan,
        "monthly_price": plans[plan],
        "contract_type": contract_type,
        "auto_renew": auto_renew
    }

    subscriptions.append({
        "subscription_id": f"SUB{cid[1:]}",
        "customer_id": cid,
        "plan_name": plan,
        "monthly_price": plans[plan],
        "contract_type": contract_type,
        "auto_renew": auto_renew
    })


subscriptions_df = pd.DataFrame(
    subscriptions
)


# ==========================================
# 4. GENERATE USAGE DATA
# ==========================================

print("Generating usage data...")

usage_records = []

usage_metrics = {}

usage_id = 1

devices = [
    "Mobile",
    "Web",
    "Desktop"
]

for cid in customer_ids:

    number_of_days = random.randint(
        5,
        30
    )

    total_logins = 0
    total_session_minutes = 0
    total_features_used = 0

    for _ in range(number_of_days):

        logins = random.randint(
            0,
            5
        )

        session_minutes = random.randint(
            0,
            120
        )

        features_used = random.randint(
            0,
            10
        )

        total_logins += logins
        total_session_minutes += session_minutes
        total_features_used += features_used

        usage_records.append({
            "usage_id": f"U{usage_id}",
            "customer_id": cid,
            "usage_date": fake.date_between(
                start_date="-90d",
                end_date="today"
            ),
            "login_count": logins,
            "session_minutes": session_minutes,
            "features_used": features_used,
            "device_type": random.choice(
                devices
            )
        })

        usage_id += 1

    usage_metrics[cid] = {
        "avg_logins": total_logins / number_of_days,
        "avg_session_minutes":
            total_session_minutes / number_of_days,
        "avg_features_used":
            total_features_used / number_of_days
    }


usage_df = pd.DataFrame(
    usage_records
)


# ==========================================
# 5. GENERATE SUPPORT TICKETS
# ==========================================

print("Generating support tickets...")

tickets = []

support_metrics = {}

ticket_id = 1

issue_categories = [
    "Billing",
    "Technical",
    "Login",
    "Subscription",
    "Payment"
]

priorities = [
    "Low",
    "Medium",
    "High"
]

for cid in customer_ids:

    number_of_tickets = random.randint(
        0,
        5
    )

    negative_count = 0

    for _ in range(number_of_tickets):

        resolution_hours = random.randint(
            1,
            72
        )

        if resolution_hours > 36:
            sentiment = "Negative"

        elif resolution_hours > 12:
            sentiment = "Neutral"

        else:
            sentiment = "Positive"

        if sentiment == "Negative":
            negative_count += 1

        tickets.append({
            "ticket_id": f"S{ticket_id}",
            "customer_id": cid,
            "ticket_date": fake.date_between(
                start_date="-1y",
                end_date="today"
            ),
            "issue_category": random.choice(
                issue_categories
            ),
            "priority": random.choice(
                priorities
            ),
            "resolution_hours": resolution_hours,
            "ticket_status": random.choice([
                "Resolved",
                "Resolved",
                "Open"
            ]),
            "customer_sentiment": sentiment
        })

        ticket_id += 1

    support_metrics[cid] = {
        "total_tickets": number_of_tickets,
        "negative_tickets": negative_count
    }


tickets_df = pd.DataFrame(
    tickets
)


# ==========================================
# 6. GENERATE CHURN
# ==========================================

print("Generating realistic churn labels...")

churn_data = []

churn_lookup = {}

for cid in customer_ids:

    # Base churn probability
    probability = 0.05

    avg_logins = usage_metrics[cid][
        "avg_logins"
    ]

    negative_tickets = support_metrics[cid][
        "negative_tickets"
    ]

    total_tickets = support_metrics[cid][
        "total_tickets"
    ]

    contract_type = subscription_info[cid][
        "contract_type"
    ]

    auto_renew = subscription_info[cid][
        "auto_renew"
    ]

    # -------------------------
    # LOW ENGAGEMENT
    # -------------------------

    if avg_logins < 1:
        probability += 0.25

    elif avg_logins < 2:
        probability += 0.10

    # -------------------------
    # SUPPORT PROBLEMS
    # -------------------------

    if negative_tickets >= 2:
        probability += 0.30

    elif total_tickets >= 4:
        probability += 0.15

    # -------------------------
    # CONTRACT TYPE
    # -------------------------

    if contract_type == "Monthly":
        probability += 0.10

    # -------------------------
    # AUTO RENEWAL
    # -------------------------

    if auto_renew == "No":
        probability += 0.08

    # -------------------------
    # RANDOM BUSINESS FACTORS
    # -------------------------

    probability += np.random.uniform(
        -0.05,
        0.05
    )

    # Keep probability valid
    probability = max(
        0.01,
        min(probability, 0.95)
    )

    # Generate actual churn
    churned = np.random.choice(
        [0, 1],
        p=[
            1 - probability,
            probability
        ]
    )

    churn_lookup[cid] = churned

    churn_data.append({
        "customer_id": cid,
        "churn_probability":
            round(probability, 3),
        "churned": churned
    })


churn_df = pd.DataFrame(
    churn_data
)

customers_df = customers_df.merge(
    churn_df,
    on="customer_id",
    how="left"
)


# ==========================================
# 7. UPDATE CUSTOMER STATUS
# ==========================================

customers_df.loc[
    customers_df["churned"] == 1,
    "customer_status"
] = "Churned"


# ==========================================
# 8. GENERATE TRANSACTIONS
# ==========================================

print("Generating transactions...")

transactions = []

transaction_id = 1

payment_methods = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Net Banking"
]

product_categories = [
    "Subscription",
    "Add-on",
    "Premium Service"
]

for cid in customer_ids:

    is_churned = churn_lookup[cid]

    # Churned customers generally
    # have fewer transactions
    if is_churned == 1:
        number_of_transactions = random.randint(
            1,
            5
        )

    else:
        number_of_transactions = random.randint(
            5,
            15
        )

    for _ in range(
        number_of_transactions
    ):

        transactions.append({
            "transaction_id":
                f"T{transaction_id}",

            "customer_id":
                cid,

            "transaction_date":
                fake.date_between(
                    start_date="-1y",
                    end_date="today"
                ),

            "amount":
                random.randint(
                    300,
                    3000
                ),

            "payment_method":
                random.choice(
                    payment_methods
                ),

            "transaction_status":
                random.choice([
                    "Success",
                    "Success",
                    "Success",
                    "Failed"
                ]),

            "product_category":
                random.choice(
                    product_categories
                )
        })

        transaction_id += 1


transactions_df = pd.DataFrame(
    transactions
)


# ==========================================
# 9. EXPORT DATASETS
# ==========================================

print("Saving CSV files...")

customers_df.to_csv(
    DATA_DIR / "customers.csv",
    index=False
)

subscriptions_df.to_csv(
    DATA_DIR / "subscriptions.csv",
    index=False
)

usage_df.to_csv(
    DATA_DIR / "usage_data.csv",
    index=False
)

tickets_df.to_csv(
    DATA_DIR / "support_tickets.csv",
    index=False
)

transactions_df.to_csv(
    DATA_DIR / "transactions.csv",
    index=False
)


# ==========================================
# 10. SUMMARY
# ==========================================

print("\n" + "=" * 50)
print("DATA GENERATION COMPLETED")
print("=" * 50)

print(
    f"Customers: {len(customers_df):,}"
)

print(
    f"Subscriptions: {len(subscriptions_df):,}"
)

print(
    f"Usage Records: {len(usage_df):,}"
)

print(
    f"Support Tickets: {len(tickets_df):,}"
)

print(
    f"Transactions: {len(transactions_df):,}"
)

print(
    f"\nChurn Rate: "
    f"{customers_df['churned'].mean() * 100:.2f}%"
)

print(
    f"\nFiles saved in:\n{DATA_DIR}"
)

print(
    "\nAll 5 datasets generated successfully!"
)