# Customer Churn Prediction with Business Action Layer
# Technical CSM Portfolio Project
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================
# STEP 1: DATA LOADING AND EXPLORATION
# ============================================

print("=" * 60)
print("CUSTOMER CHURN ANALYSIS - TECHNICAL CSM APPROACH")
print("=" * 60)

# Load the dataset
# Note: Replace 'WA_Fn-UseC_-Telco-Customer-Churn.csv' with your actual file path
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("\n📊 Dataset Overview:")
print(f"Total Customers: {len(df)}")
print(f"Features: {df.shape[1]}")
print(f"\nFirst few rows:")
print(df.head())

print("\n📋 Data Types:")
print(df.dtypes)

print("\n🔍 Missing Values:")
print(df.isnull().sum())

# ============================================
# STEP 2: DATA CLEANING AND PREPROCESSING
# ============================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Create a copy for processing
df_clean = df.copy()

# Handle TotalCharges - it's sometimes stored as object instead of numeric
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with median (typically new customers)
median_charges = df_clean['TotalCharges'].median()
df_clean['TotalCharges'].fillna(median_charges, inplace=True)
print(f"✓ Filled {df['TotalCharges'].isnull().sum()} missing TotalCharges values with median: ${median_charges:.2f}")

# Convert Churn to binary (Yes=1, No=0)
df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})

# Identify categorical and numerical columns
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('customerID')  # Remove ID column

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

print(f"\n✓ Categorical features: {len(categorical_cols)}")
print(f"✓ Numerical features: {len(numerical_cols)}")

# Convert categorical variables to numeric using Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le
    
print(f"✓ Converted {len(categorical_cols)} categorical variables to numeric")

# ============================================
# STEP 3: FEATURE ENGINEERING (CSM PERSPECTIVE)
# ============================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING - BUSINESS METRICS")
print("=" * 60)

# Calculate Customer Lifetime Value (CLV) estimate
df_clean['CLV_Estimate'] = df_clean['MonthlyCharges'] * df_clean['tenure']

# Create customer value segments
df_clean['ValueSegment'] = pd.qcut(df_clean['MonthlyCharges'], 
                                    q=3, 
                                    labels=['Low', 'Medium', 'High'])

# Calculate revenue at risk
df_clean['MonthlyRevenueAtRisk'] = df_clean['MonthlyCharges'] * df_clean['Churn']

print("✓ Created CLV_Estimate feature")
print("✓ Created ValueSegment feature (Low/Medium/High)")
print("✓ Calculated MonthlyRevenueAtRisk")

# ============================================
# STEP 4: PREPARE DATA FOR MODELING
# ============================================

print("\n" + "=" * 60)
print("PREPARING DATA FOR MODELING")
print("=" * 60)

# Select features for modeling (exclude ID and engineered features not for training)
feature_cols = [col for col in df_clean.columns 
                if col not in ['customerID', 'Churn', 'MonthlyRevenueAtRisk', 
                              'CLV_Estimate', 'ValueSegment']]

X = df_clean[feature_cols]
y = df_clean['Churn']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                      test_size=0.2, 
                                                      random_state=42,
                                                      stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Training set: {len(X_train)} customers")
print(f"✓ Test set: {len(X_test)} customers")
print(f"✓ Churn rate in training: {y_train.mean():.2%}")
print(f"✓ Churn rate in test: {y_test.mean():.2%}")

# ============================================
# STEP 5: MODEL TRAINING
# ============================================

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# Model 1: Logistic Regression
print("\n🔹 Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

print("✓ Logistic Regression trained")
print(f"  Training accuracy: {lr_model.score(X_train_scaled, y_train):.2%}")
print(f"  Test accuracy: {lr_model.score(X_test_scaled, y_test):.2%}")
print(f"  ROC-AUC Score: {roc_auc_score(y_test, lr_pred_proba):.3f}")

# Model 2: Random Forest
print("\n🔹 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print("✓ Random Forest trained")
print(f"  Training accuracy: {rf_model.score(X_train, y_train):.2%}")
print(f"  Test accuracy: {rf_model.score(X_test, y_test):.2%}")
print(f"  ROC-AUC Score: {roc_auc_score(y_test, rf_pred_proba):.3f}")

# ============================================
# STEP 6: THE CS TWIST - BUSINESS ACTION LAYER
# ============================================

print("\n" + "=" * 60)
print("🎯 BUSINESS ACTION LAYER (CS TWIST)")
print("=" * 60)

# Use Random Forest predictions (typically more accurate)
test_df = df_clean.loc[X_test.index].copy()
test_df['ChurnProbability'] = rf_pred_proba
test_df['PredictedChurn'] = rf_pred

# Define business action thresholds
# CS BUSINESS RULE: Segment customers by churn risk and assign actions
def assign_cs_action(prob):
    """
    CS Business Logic: Assign customer success actions based on churn probability
    
    - Probability > 0.7: CRITICAL - Trigger Executive Business Review (EBR)
    - Probability 0.5-0.7: HIGH RISK - Assign dedicated CSM intervention
    - Probability 0.3-0.5: MEDIUM RISK - Automated health check + outreach
    - Probability < 0.3: LOW RISK - Standard engagement model
    """
    if prob > 0.7:
        return 'CRITICAL: Executive Business Review'
    elif prob > 0.5:
        return 'HIGH: Dedicated CSM Intervention'
    elif prob > 0.3:
        return 'MEDIUM: Health Check + Outreach'
    else:
        return 'LOW: Standard Engagement'

test_df['CS_Action'] = test_df['ChurnProbability'].apply(assign_cs_action)

# Calculate business impact metrics
print("\n💰 BUSINESS IMPACT ANALYSIS:")
print("-" * 60)

# Revenue at risk by segment
revenue_at_risk = test_df[test_df['ChurnProbability'] > 0.5]['MonthlyCharges'].sum()
safe_revenue = test_df[test_df['ChurnProbability'] <= 0.5]['MonthlyCharges'].sum()
total_revenue = test_df['MonthlyCharges'].sum()

print(f"Total Monthly Revenue: ${total_revenue:,.2f}")
print(f"Revenue at Risk (>50% churn prob): ${revenue_at_risk:,.2f} ({revenue_at_risk/total_revenue:.1%})")
print(f"Safe Revenue (≤50% churn prob): ${safe_revenue:,.2f} ({safe_revenue/total_revenue:.1%})")

# CS Action Summary
print("\n📋 CS ACTION PLAN:")
print("-" * 60)
action_summary = test_df['CS_Action'].value_counts().sort_index()
for action, count in action_summary.items():
    pct = count / len(test_df) * 100
    action_revenue = test_df[test_df['CS_Action'] == action]['MonthlyCharges'].sum()
    print(f"{action}: {count} customers ({pct:.1f}%) - ${action_revenue:,.2f}/month")

# High-value customers at risk (CSM priority targets)
print("\n🎯 HIGH-PRIORITY TARGETS:")
print("-" * 60)
high_value_at_risk = test_df[
    (test_df['ChurnProbability'] > 0.5) & 
    (test_df['MonthlyCharges'] > test_df['MonthlyCharges'].median())
].sort_values('ChurnProbability', ascending=False)

print(f"High-value customers at risk: {len(high_value_at_risk)}")
print(f"Revenue impact: ${high_value_at_risk['MonthlyCharges'].sum():,.2f}/month")
print(f"Avg churn probability: {high_value_at_risk['ChurnProbability'].mean():.1%}")

# ============================================
# STEP 7: VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("CREATING BUSINESS VISUALIZATIONS")
print("=" * 60)

# Visualization 1: Revenue at Risk vs Safe Revenue
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Chart 1: Revenue at Risk Breakdown
revenue_data = pd.DataFrame({
    'Category': ['Revenue at Risk\n(>50% churn)', 'Safe Revenue\n(≤50% churn)'],
    'Amount': [revenue_at_risk, safe_revenue]
})
colors = ['#E74C3C', '#2ECC71']
axes[0, 0].bar(revenue_data['Category'], revenue_data['Amount'], color=colors, alpha=0.7)
axes[0, 0].set_title('Monthly Revenue: At Risk vs Safe', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Monthly Revenue ($)', fontsize=12)
for i, v in enumerate(revenue_data['Amount']):
    axes[0, 0].text(i, v, f'${v:,.0f}\n({v/total_revenue:.1%})', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

# Chart 2: CS Action Distribution
action_counts = test_df['CS_Action'].value_counts()
colors_action = ['#C0392B', '#E67E22', '#F39C12', '#27AE60']
axes[0, 1].barh(action_counts.index, action_counts.values, color=colors_action, alpha=0.7)
axes[0, 1].set_title('Customer Distribution by CS Action Priority', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Number of Customers', fontsize=12)
for i, v in enumerate(action_counts.values):
    axes[0, 1].text(v, i, f' {v} ({v/len(test_df)*100:.1f}%)', 
                    va='center', fontsize=10)

# Chart 3: Churn Probability Distribution
axes[1, 0].hist(test_df['ChurnProbability'], bins=50, color='#3498DB', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(0.7, color='red', linestyle='--', linewidth=2, label='Critical Threshold (0.7)')
axes[1, 0].axvline(0.5, color='orange', linestyle='--', linewidth=2, label='High Risk Threshold (0.5)')
axes[1, 0].set_title('Churn Probability Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Churn Probability', fontsize=12)
axes[1, 0].set_ylabel('Number of Customers', fontsize=12)
axes[1, 0].legend()

# Chart 4: Feature Importance (Random Forest)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

axes[1, 1].barh(feature_importance['Feature'], feature_importance['Importance'], 
                color='#9B59B6', alpha=0.7)
axes[1, 1].set_title('Top 10 Churn Drivers (Feature Importance)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Importance Score', fontsize=12)
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('churn_business_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: churn_business_analysis.png")
plt.show()

# ============================================
# STEP 8: EXPORT RESULTS FOR CS TEAM
# ============================================

print("\n" + "=" * 60)
print("EXPORTING RESULTS")
print("=" * 60)

# Export high-priority customer list for CS team
export_df = test_df[['ChurnProbability', 'CS_Action', 'MonthlyCharges', 
                      'tenure', 'CLV_Estimate']].copy()
export_df = export_df.sort_values('ChurnProbability', ascending=False)

# Save to CSV
export_df.to_csv('cs_action_plan.csv', index=False)
print("✓ Exported CS Action Plan: cs_action_plan.csv")

# Create summary report
with open('churn_analysis_summary.txt', 'w') as f:
    f.write("CUSTOMER CHURN ANALYSIS - EXECUTIVE SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
    f.write(f"Total Customers Analyzed: {len(test_df)}\n\n")
    f.write(f"MODEL PERFORMANCE:\n")
    f.write(f"  Random Forest ROC-AUC: {roc_auc_score(y_test, rf_pred_proba):.3f}\n\n")
    f.write(f"BUSINESS IMPACT:\n")
    f.write(f"  Total Monthly Revenue: ${total_revenue:,.2f}\n")
    f.write(f"  Revenue at Risk: ${revenue_at_risk:,.2f} ({revenue_at_risk/total_revenue:.1%})\n")
    f.write(f"  Safe Revenue: ${safe_revenue:,.2f}\n\n")
    f.write(f"CS ACTION PLAN:\n")
    for action, count in action_summary.items():
        f.write(f"  {action}: {count} customers\n")

print("✓ Exported summary report: churn_analysis_summary.txt")

print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE!")
print("=" * 60)
print("\nNext Steps for CS Team:")
print("1. Review cs_action_plan.csv for prioritized customer list")
print("2. Schedule EBRs for customers with >70% churn probability")
print("3. Assign CSMs to high-value at-risk customers")
print("4. Monitor health check automation for medium-risk customers")