###EDA

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("synthetic_transactions.csv")

# Display first few rows
print(" First 5 rows of dataset:\n")
print(df.head())

# Dataset summary
print("Dataset Info:\n")
print(df.info())

# Check for missing values
print("Missing Values:\n")
print(df.isnull().sum())

# Check basic statistics
print(" Statistical Summary:\n")
print(df.describe())

plt.figure(figsize=(6,4))
sns.countplot(x=df["is_fraud"], palette="coolwarm")
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Is Fraud (1 = Fraud, 0 = Not Fraud)")
plt.ylabel("Count")
plt.show()

fraud_distribution = df["is_fraud"].value_counts(normalize=True)
print("\n🔹 Class Distribution:")
print(fraud_distribution)


# Feature Distributions

df[['amount', 'gas_fee', 'transaction_count', 'wallet_age']].hist(bins=30, figsize=(12, 8), color="steelblue")
plt.suptitle("Feature Distributions")
plt.show()




# #Outlier Analysis using Boxplots

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["is_fraud"], y=df["amount"], palette="coolwarm")
plt.title("Transaction Amount by Fraud Status")
plt.xlabel("Is Fraud")
plt.ylabel("Transaction Amount")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["is_fraud"], y=df["gas_fee"], palette="coolwarm")
plt.title("Gas Fee by Fraud Status")
plt.xlabel("Is Fraud")
plt.ylabel("Gas Fee")
plt.show()


### Anomaly Detection with IsolationForest

# Select features
features = df[['amount', 'gas_fee', 'transaction_count', 'wallet_age']]

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train IsolationForest model
clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
clf.fit(features_scaled)

# anomaly scores
df["anomaly_score"] = clf.decision_function(features_scaled)
df["isolation_forest_label"] = clf.predict(features_scaled)

df["isolation_forest_label"] = df["isolation_forest_label"].apply(lambda x: 1 if x == -1 else 0)

#Anomaly Detection
plt.figure(figsize=(6,4))
sns.countplot(x=df["isolation_forest_label"], palette="coolwarm")
plt.title("Isolation Forest Predicted Fraud Cases")
plt.xlabel("Predicted Fraud (1 = Fraud, 0 = Not Fraud)")
plt.ylabel("Count")
plt.show()

print("\nTop 10 Fraudulent Transactions by Anomaly Score:")
print(df[["amount", "gas_fee", "transaction_count", "wallet_age", "is_fraud", "anomaly_score"]].sort_values(by="anomaly_score").head(10))

df.to_csv("processed_fraud_data.csv", index=False)
print("\n✅ Processed data saved as 'processed_fraud_data.csv'")

