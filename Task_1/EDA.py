import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better visualizations
plt.style.use('fivethirtyeight')
sns.set_palette("husl")

# Read and prepare data
transactions_df = pd.read_csv('Transactions.csv')
products_df = pd.read_csv('Products.csv')
customers_df = pd.read_csv('Customers.csv')

transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])

# 1. Customer Purchase Patterns & Loyalty Visualization
def visualize_customer_segments():
    plt.figure(figsize=(15, 10))

    # Customer RFM Analysis
    customer_metrics = transactions_df.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (x.max() - x.min()).days,  # Frequency
        'TotalValue': ['count', 'sum']  # Recency and Monetary
    }).reset_index()

    customer_metrics.columns = ['CustomerID', 'Days_Active', 'Transaction_Count', 'Total_Spent']

    # Create subplot for customer segments
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=customer_metrics, x='Transaction_Count', y='Total_Spent', alpha=0.5)
    plt.title('Customer Segments by Transaction Count and Total Spend')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Total Spend ($)')

    # Customer Spend Distribution
    plt.subplot(2, 2, 2)
    sns.histplot(data=customer_metrics, x='Total_Spent', bins=30)
    plt.title('Distribution of Customer Total Spend')
    plt.xlabel('Total Spend ($)')

    plt.tight_layout()
    plt.savefig('customer_segments.png')
    plt.close()

# 2. Product Category Performance Visualization
def visualize_product_performance():
    plt.figure(figsize=(15, 10))

    # Merge transactions with products
    product_sales = pd.merge(transactions_df, products_df, on='ProductID')
    category_performance = product_sales.groupby('Category').agg({
        'TotalValue': 'sum',
        'Quantity': 'sum',
        'TransactionID': 'count'
    }).reset_index()

    # Category Revenue
    plt.subplot(2, 2, 1)
    sns.barplot(data=category_performance, x='Category', y='TotalValue')
    plt.title('Revenue by Product Category')
    plt.xticks(rotation=45)
    plt.ylabel('Total Revenue ($)')

    # Category Units Sold
    plt.subplot(2, 2, 2)
    sns.barplot(data=category_performance, x='Category', y='Quantity')
    plt.title('Units Sold by Product Category')
    plt.xticks(rotation=45)
    plt.ylabel('Total Units Sold')

    plt.tight_layout()
    plt.savefig('product_performance.png')
    plt.close()

# 3. Regional Sales Distribution Visualization
def visualize_regional_performance():
    plt.figure(figsize=(15, 10))

    # Merge transactions with customers
    customer_sales = pd.merge(transactions_df, customers_df, on='CustomerID')
    regional_performance = customer_sales.groupby('Region').agg({
        'TotalValue': ['sum', 'mean'],
        'TransactionID': 'count'
    }).reset_index()

    regional_performance.columns = ['Region', 'Total_Revenue', 'Avg_Transaction', 'Transaction_Count']

    # Regional Revenue
    plt.subplot(2, 2, 1)
    sns.barplot(data=regional_performance, x='Region', y='Total_Revenue')
    plt.title('Total Revenue by Region')
    plt.xticks(rotation=45)
    plt.ylabel('Total Revenue ($)')

    # Regional Average Transaction
    plt.subplot(2, 2, 2)
    sns.barplot(data=regional_performance, x='Region', y='Avg_Transaction')
    plt.title('Average Transaction Value by Region')
    plt.xticks(rotation=45)
    plt.ylabel('Average Transaction Value ($)')

    plt.tight_layout()
    plt.savefig('regional_performance.png')
    plt.close()

# 4. Temporal Sales Patterns Visualization
def visualize_temporal_patterns():
    plt.figure(figsize=(15, 10))

    # Monthly Revenue Trend
    monthly_revenue = transactions_df.groupby(transactions_df['TransactionDate'].dt.to_period('M'))\
        .agg({'TotalValue': 'sum'}).reset_index()
    monthly_revenue['TransactionDate'] = monthly_revenue['TransactionDate'].astype(str)

    plt.subplot(2, 1, 1)
    sns.lineplot(data=monthly_revenue, x='TransactionDate', y='TotalValue')
    plt.title('Monthly Revenue Trend')
    plt.xticks(rotation=45)
    plt.ylabel('Total Revenue ($)')

    # Daily Transaction Count
    daily_transactions = transactions_df.groupby(transactions_df['TransactionDate'].dt.date)\
        .size().reset_index(name='Transaction_Count')

    plt.subplot(2, 1, 2)
    sns.lineplot(data=daily_transactions, x='TransactionDate', y='Transaction_Count')
    plt.title('Daily Transaction Count')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Transactions')

    plt.tight_layout()
    plt.savefig('temporal_patterns.png')
    plt.close()

# 5. Product Bundle Analysis Visualization
def visualize_product_bundles():
    plt.figure(figsize=(15, 10))

    # Get products frequently bought together
    transaction_products = transactions_df.groupby(['TransactionID', 'ProductID'])['Quantity'].sum().unstack().fillna(0)
    product_correlations = transaction_products.corr()

    # Product Correlation Heatmap
    sns.heatmap(product_correlations, cmap='coolwarm', center=0)
    plt.title('Product Purchase Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('product_bundles.png')
    plt.close()

# Generate all visualizations
def generate_all_visualizations():
    print("Generating visualizations...")
    visualize_customer_segments()
    visualize_product_performance()
    visualize_regional_performance()
    visualize_temporal_patterns()
    visualize_product_bundles()
    print("All visualizations generated!")

    # Calculate and print key metrics
    print("\nKey Business Metrics:")
    print(f"Total Revenue: ${transactions_df['TotalValue'].sum():,.2f}")
    print(f"Average Transaction Value: ${transactions_df['TotalValue'].mean():,.2f}")
    print(f"Total Number of Transactions: {len(transactions_df):,}")
    print(f"Total Number of Unique Customers: {transactions_df['CustomerID'].nunique():,}")
    print(f"Total Number of Products: {products_df['ProductID'].nunique():,}")

# Execute the analysis
if __name__ == "__main__":
    generate_all_visualizations()