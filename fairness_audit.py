import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# --- 1. Load the COMPAS Dataset ---
# Using the direct raw link from ProPublica's repository
url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
print("Loading COMPAS dataset...")
df = pd.read_csv(url)

# --- 2. Data Cleaning & Preprocessing ---
# We focus on the core attributes: Age, Charge Degree, Race, Sex, Priors
# Target: 'two_year_recid' (1 = re-offended within 2 years, 0 = did not)
columns = ['age', 'c_charge_degree', 'race', 'sex', 'priors_count', 'two_year_recid']
df = df[columns].copy()

# Filter for only African-American and Caucasian for direct comparison
df = df[df['race'].isin(['African-American', 'Caucasian'])]

# Encoding categorical variables
df['sex'] = df['sex'].map({'Female': 1, 'Male': 0})
df['race_code'] = df['race'].map({'Caucasian': 1, 'African-American': 0})
df['c_charge_degree'] = df['c_charge_degree'].map({'F': 1, 'M': 0}) # F=Felony, M=Misdemeanor

# Split into Features (X) and Target (y)
X = df[['age', 'c_charge_degree', 'sex', 'priors_count', 'race_code']]
y = df['two_year_recid']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 3. Train a Simple Proxy Model ---
# We simulate the risk assessment tool using Logistic Regression
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 4. Fairness Audit ---
# We will check for Disparate Impact in False Positive Rates (FPR)
# FPR = Labeling someone 'High Risk' when they did NOT re-offend.

test_df = X_test.copy()
test_df['actual'] = y_test
test_df['predicted'] = y_pred
test_df['race_label'] = test_df['race_code'].map({1: 'Caucasian', 0: 'African-American'})

def calculate_fpr(group_df):
    cm = confusion_matrix(group_df['actual'], group_df['predicted'])
    # confusion matrix layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (tn + fp)
    return fpr

# Calculate FPR for both groups
aa_data = test_df[test_df['race_label'] == 'African-American']
c_data = test_df[test_df['race_label'] == 'Caucasian']

fpr_aa = calculate_fpr(aa_data)
fpr_c = calculate_fpr(c_data)

print(f"\n--- AUDIT RESULTS ---")
print(f"False Positive Rate (African-American): {fpr_aa:.2%}")
print(f"False Positive Rate (Caucasian):        {fpr_c:.2%}")
print(f"Disparity: African-Americans are {fpr_aa/fpr_c:.2f}x more likely to be falsely flagged.")

# --- 5. Visualization ---
groups = ['African-American', 'Caucasian']
fprs = [fpr_aa, fpr_c]

plt.figure(figsize=(8, 6))
bars = plt.bar(groups, fprs, color=['#d62728', '#1f77b4'])
plt.title('Bias Audit: False Positive Rates by Race')
plt.ylabel('False Positive Rate (Wrongly Accused)')
plt.ylim(0, 0.6)

# Add text labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.1%}", ha='center', fontweight='bold')

plt.savefig('bias_audit_chart.png')
print("\nChart saved as 'bias_audit_chart.png'")