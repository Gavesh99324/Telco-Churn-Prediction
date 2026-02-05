-- Database schema
CREATE TABLE IF NOT EXISTS churn_predictions (
    id INTEGER PRIMARY KEY,
    customer_id VARCHAR(50),
    prediction INTEGER,
    probability FLOAT,
    timestamp DATETIME
);

CREATE TABLE IF NOT EXISTS churn_analytics (
    id INTEGER PRIMARY KEY,
    metric_name VARCHAR(100),
    metric_value FLOAT,
    timestamp DATETIME
);
