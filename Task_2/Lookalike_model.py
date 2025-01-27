import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json

def load_data():
    """Load and preprocess the data files"""
    transactions = pd.read_csv('Transactions.csv')
    products = pd.read_csv('Products.csv')
    customers = pd.read_csv('Customers.csv')
    return transactions, products, customers

def create_customer_features(transactions, products, customers):
    """Create feature matrix for customers"""
    # 1. Transaction Features
    transaction_features = transactions.groupby('CustomerID').agg({
        'TransactionID': 'count',
        'TotalValue': ['sum', 'mean'],
        'Quantity': ['sum', 'mean'],
        'ProductID': 'nunique'
    }).reset_index()

    # Flatten column names
    transaction_features.columns = ['CustomerID', 'transaction_count', 'total_value',
                                  'avg_transaction_value', 'total_quantity',
                                  'avg_quantity', 'unique_products']

    # 2. Product Category Preferences
    # Merge transactions with products to get categories
    txn_with_categories = transactions.merge(products[['ProductID', 'Category']],
                                           on='ProductID')

    # Calculate spending by category for each customer
    category_pivot = pd.pivot_table(
        txn_with_categories,
        values='TotalValue',
        index='CustomerID',
        columns='Category',
        aggfunc='sum',
        fill_value=0
    )

    # Convert to percentage of total spend
    category_percentages = category_pivot.div(category_pivot.sum(axis=1), axis=0)
    category_percentages.columns = [f'category_perc_{col}' for col in category_percentages.columns]

    # 3. Regional Features (One-hot encoding)
    region_dummies = pd.get_dummies(customers[['CustomerID', 'Region']],
                                  prefix='region',
                                  columns=['Region'])

    # Combine all features
    customer_features = transaction_features.merge(
        category_percentages.reset_index(), on='CustomerID', how='left'
    ).merge(
        region_dummies, on='CustomerID', how='left'
    )

    return customer_features

def calculate_similarity_scores(feature_matrix, customer_id):
    """Calculate similarity scores for a given customer"""
    # Get the customer's feature vector
    customer_vector = feature_matrix[feature_matrix['CustomerID'] == customer_id].iloc[:, 1:]
    all_vectors = feature_matrix.iloc[:, 1:]

    # Calculate cosine similarity
    similarities = cosine_similarity(customer_vector, all_vectors)[0]

    # Create DataFrame with similarities
    similarity_df = pd.DataFrame({
        'CustomerID': feature_matrix['CustomerID'],
        'similarity_score': similarities
    })

    # Remove the customer itself and sort by similarity
    similarity_df = similarity_df[similarity_df['CustomerID'] != customer_id]
    similarity_df = similarity_df.sort_values('similarity_score', ascending=False)

    return similarity_df.head(3)

def create_lookalike_model():
    """Main function to create lookalike model and generate output file"""
    print("Loading data...")
    transactions, products, customers = load_data()

    print("Creating customer features...")
    customer_features = create_customer_features(transactions, products, customers)

    # Normalize features
    print("Normalizing features...")
    feature_cols = customer_features.columns.difference(['CustomerID'])
    scaler = StandardScaler()
    customer_features[feature_cols] = scaler.fit_transform(customer_features[feature_cols])

    # Generate recommendations for customers C0001-C0020
    print("Generating recommendations...")
    recommendations = {}
    for i in range(1, 21):
        customer_id = f'C{str(i).zfill(4)}'
        similar_customers = calculate_similarity_scores(customer_features, customer_id)

        # Format recommendations
        recommendations[customer_id] = [
            {
                'customer_id': row['CustomerID'],
                'similarity_score': float(row['similarity_score'])
            }
            for _, row in similar_customers.iterrows()
        ]

        # Print progress
        print(f"Processed customer {customer_id}")

    # Save to CSV
    print("Saving results...")
    with open('FirstName_LastName_Lookalike.csv', 'w') as f:
        f.write('CustomerID,Recommendations\n')
        for customer_id, recs in recommendations.items():
            recs_str = json.dumps(recs)
            f.write(f'{customer_id},{recs_str}\n')

    # Print sample results
    print("\nSample recommendations for first 3 customers:")
    for i in range(1, 4):
        customer_id = f'C{str(i).zfill(4)}'
        print(f"\nCustomer {customer_id}:")
        for rec in recommendations[customer_id]:
            print(f"- {rec['customer_id']}: {rec['similarity_score']:.4f}")

if __name__ == "__main__":
    create_lookalike_model()
    print("\nLookalike model completed successfully!")