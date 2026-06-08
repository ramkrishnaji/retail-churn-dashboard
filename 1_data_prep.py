"""
Step 1: Data Preparation & RFM Feature Engineering
Author: Student
Course: Retail Analytics Project

This script prepares the raw Olist Brazilian E-commerce dataset for clustering and churn prediction.
Steps involved:
1. Define folder structures (data/raw, data/processed, models, reports).
2. Copy downloaded raw datasets from cache to local project directories.
3. Merge relevant tables (orders, customers, payments, reviews).
4. Perform feature engineering at the customer level:
   - RFM (Recency, Frequency, Monetary)
   - Derived features (Average Order Value, Average Review Score, Preferred Payment Method, Cancellation Rate).
5. Clean data, impute missing values, and export the processed dataset.
"""

import os
import shutil
import pandas as pd
import numpy as np

# ----------------------------------------------------
# 1. Setup Directory Structure
# ----------------------------------------------------
print("--- Step 1: Setting up directories ---")
dirs = ['data/raw', 'data/processed', 'models', 'reports']
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created/verified directory: {d}")

# ----------------------------------------------------
# 2. Copy Raw Files from Kaggle Cache
# ----------------------------------------------------
print("\n--- Step 2: Copying raw CSV files from Kaggle cache ---")
cache_dir = r"C:\Users\ramro\.cache\kagglehub\datasets\olistbr\brazilian-ecommerce\versions\2"
raw_dir = "data/raw"

files_to_copy = [
    'olist_customers_dataset.csv',
    'olist_orders_dataset.csv',
    'olist_order_items_dataset.csv',
    'olist_order_payments_dataset.csv',
    'olist_order_reviews_dataset.csv'
]

for filename in files_to_copy:
    src_path = os.path.join(cache_dir, filename)
    dest_path = os.path.join(raw_dir, filename)
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)
        print(f"Successfully copied {filename} to {raw_dir}")
    else:
        print(f"Warning: Source file {src_path} not found!")

# ----------------------------------------------------
# 3. Load Datasets
# ----------------------------------------------------
print("\n--- Step 3: Loading raw datasets ---")
customers = pd.read_csv("data/raw/olist_customers_dataset.csv")
orders = pd.read_csv("data/raw/olist_orders_dataset.csv")
payments = pd.read_csv("data/raw/olist_order_payments_dataset.csv")
reviews = pd.read_csv("data/raw/olist_order_reviews_dataset.csv")

print(f"Customers rows: {customers.shape[0]}, cols: {customers.shape[1]}")
print(f"Orders rows: {orders.shape[0]}, cols: {orders.shape[1]}")
print(f"Payments rows: {payments.shape[0]}, cols: {payments.shape[1]}")
print(f"Reviews rows: {reviews.shape[0]}, cols: {reviews.shape[1]}")

# ----------------------------------------------------
# 4. Data Preprocessing & Merging
# ----------------------------------------------------
print("\n--- Step 4: Preprocessing and Merging ---")

# Parse dates in orders table
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

# Note: Olist has customer_id (order-specific) and customer_unique_id (actual customer).
# We need to map customer_id to customer_unique_id.
print("Merging orders and customers datasets to map unique customer IDs...")
orders_with_cust = orders.merge(customers, on='customer_id', how='inner')

# Aggregate payments by order_id (some orders have multiple payment entries/splits)
print("Aggregating order-level payment values...")
order_payments = payments.groupby('order_id')['payment_value'].sum().reset_index()

# Aggregate reviews by order_id (in case of multiple reviews for the same order)
print("Aggregating order-level review scores...")
order_reviews = reviews.groupby('order_id')['review_score'].mean().reset_index()

# Merge all order-level features back to the main customer order dataframe
print("Merging all order details...")
orders_merged = orders_with_cust.merge(order_payments, on='order_id', how='left')
orders_merged = orders_merged.merge(order_reviews, on='order_id', how='left')

# Fill missing payment values with 0 and missing reviews with global average score
orders_merged['payment_value'] = orders_merged['payment_value'].fillna(0)
global_avg_review = orders_merged['review_score'].mean()
orders_merged['review_score'] = orders_merged['review_score'].fillna(global_avg_review)

# Add is_canceled flag
orders_merged['is_canceled'] = np.where(orders_merged['order_status'] == 'canceled', 1, 0)

# Calculate preferred payment method per unique customer
print("Calculating preferred payment method per customer...")
# We merge payment type with unique customer id
cust_payment_types = orders_with_cust.merge(payments, on='order_id', how='inner')
preferred_payments = cust_payment_types.groupby('customer_unique_id')['payment_type'].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'
).reset_index().rename(columns={'payment_type': 'preferred_payment_method'})

# ----------------------------------------------------
# 5. Customer-Level Feature Engineering (RFM)
# ----------------------------------------------------
print("\n--- Step 5: Engineering RFM and Derived Features at Customer Level ---")

# Establish snapshot date: 1 day after the latest purchase in the dataset
max_purchase_date = orders_merged['order_purchase_timestamp'].max()
snapshot_date = max_purchase_date + pd.Timedelta(days=1)
print(f"Latest purchase date in dataset: {max_purchase_date}")
print(f"Using reference snapshot date: {snapshot_date}")

# Calculate Recency: Days since last purchase
print("Computing Recency...")
recency_df = orders_merged.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
recency_df['recency'] = (snapshot_date - recency_df['order_purchase_timestamp']).dt.days
recency_df = recency_df.drop(columns=['order_purchase_timestamp'])

# Calculate Frequency: Total unique orders per customer
print("Computing Frequency...")
frequency_df = orders_merged.groupby('customer_unique_id')['order_id'].nunique().reset_index().rename(columns={'order_id': 'frequency'})

# Calculate Monetary: Total spent per customer
print("Computing Monetary (Total Spend)...")
monetary_df = orders_merged.groupby('customer_unique_id')['payment_value'].sum().reset_index().rename(columns={'payment_value': 'monetary'})

# Calculate cancellation rate and average review score per customer
print("Computing Average Review Score & Cancellation Rate...")
review_cancel_df = orders_merged.groupby('customer_unique_id').agg(
    avg_review_score=('review_score', 'mean'),
    cancellation_rate=('is_canceled', 'mean')
).reset_index()

# Combine all customer metrics
print("Merging all customer-level metrics...")
customer_features = recency_df.merge(frequency_df, on='customer_unique_id', how='inner')
customer_features = customer_features.merge(monetary_df, on='customer_unique_id', how='inner')
customer_features = customer_features.merge(review_cancel_df, on='customer_unique_id', how='inner')
customer_features = customer_features.merge(preferred_payments, on='customer_unique_id', how='left')

# Fill missing preferred payment with 'unknown'
customer_features['preferred_payment_method'] = customer_features['preferred_payment_method'].fillna('unknown')

# Calculate Average Order Value (AOV)
customer_features['avg_order_value'] = customer_features['monetary'] / customer_features['frequency']

# Rearrange columns for clarity
customer_features = customer_features[[
    'customer_unique_id', 'recency', 'frequency', 'monetary',
    'avg_order_value', 'avg_review_score', 'preferred_payment_method', 'cancellation_rate'
]]

# Show head of the dataset
print("\nFirst 5 rows of engineered customer feature dataset:")
print(customer_features.head())

# Save to processed data folder
output_path = "data/processed/customer_features.csv"
customer_features.to_csv(output_path, index=False)
print(f"\nSaved processed customer features to {output_path} (Shape: {customer_features.shape})")
print("Data prep stage finished successfully!")
