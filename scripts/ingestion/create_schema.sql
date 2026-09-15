-- =============================================
-- DROP OLD TABLES
-- =============================================

DROP TABLE IF EXISTS ingestion_log CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS usage_data CASCADE;
DROP TABLE IF EXISTS subscriptions CASCADE;
DROP TABLE IF EXISTS customers CASCADE;


-- =============================================
-- INGESTION LOG TABLE
-- =============================================

CREATE TABLE ingestion_log (
    log_id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    source_file VARCHAR(255) NOT NULL,
    rows_loaded INTEGER DEFAULT 0,
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- =============================================
-- CUSTOMERS TABLE
-- =============================================

CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    age INTEGER CHECK (age >= 18 AND age <= 100),
    gender VARCHAR(20),
    city VARCHAR(100),
    country VARCHAR(100),
    signup_date DATE,
    customer_status VARCHAR(50),
    churn_probability NUMERIC(5,2),
    churned INTEGER CHECK (churned IN (0, 1))
);


-- =============================================
-- SUBSCRIPTIONS TABLE
-- =============================================

CREATE TABLE subscriptions (
    subscription_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    plan_name VARCHAR(50),
    monthly_price NUMERIC(10,2),
    contract_type VARCHAR(20),
    auto_renew VARCHAR(10),

    CONSTRAINT fk_subscription_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
);


-- =============================================
-- USAGE DATA TABLE
-- =============================================

CREATE TABLE usage_data (
    usage_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    usage_date DATE,
    login_count INTEGER CHECK (login_count >= 0),
    session_minutes INTEGER CHECK (session_minutes >= 0),
    features_used INTEGER CHECK (features_used >= 0),
    device_type VARCHAR(50),

    CONSTRAINT fk_usage_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
);


-- =============================================
-- SUPPORT TICKETS TABLE
-- =============================================

CREATE TABLE support_tickets (
    ticket_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    ticket_date DATE,
    issue_category VARCHAR(100),
    priority VARCHAR(20),
    resolution_hours INTEGER CHECK (resolution_hours >= 0),
    ticket_status VARCHAR(50),
    customer_sentiment VARCHAR(20),

    CONSTRAINT fk_support_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
);


-- =============================================
-- TRANSACTIONS TABLE
-- =============================================

CREATE TABLE transactions (
    transaction_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    transaction_date DATE,
    amount NUMERIC(12,2) CHECK (amount >= 0),
    payment_method VARCHAR(50),
    transaction_status VARCHAR(50),
    product_category VARCHAR(100),

    CONSTRAINT fk_transaction_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
);