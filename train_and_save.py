# Training script extracted from the notebook to produce artifacts/model.pkl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

np.random.seed(42)

n = 5000
age = np.random.normal(40, 12, n).clip(18, 80).astype(int)
income = np.random.lognormal(10.5, 0.7, n).astype(int)
loan_amount = np.random.normal(15000, 8000, n).clip(1000, 100000).astype(int)
credit_card_debt = np.random.normal(3000, 2500, n).clip(0, 50000).astype(int)
payment_history = np.random.choice(['Excellent','Good','Late','Default'], size=n, p=[0.5,0.3,0.15,0.05])
num_loans = np.random.poisson(1.5, n)
credit_util = np.random.beta(2,5, n)
employment_status = np.random.choice(['Employed','Self-employed','Unemployed','Retired'], size=n, p=[0.7,0.15,0.1,0.05])
existing_debts = np.random.normal(10000, 7000, n).clip(0, 100000).astype(int)
savings = np.random.normal(8000, 10000, n).clip(0, 200000).astype(int)

score = (income/1000)*0.3 + (savings/1000)*0.2 - (loan_amount/1000)*0.25 - (existing_debts/1000)*0.15 + (num_loans * -1.0)
hist_factor = np.array([1.2 if s=='Excellent' else 1.0 if s=='Good' else 0.7 if s=='Late' else 0.4 for s in payment_history])
score = score * hist_factor + np.random.normal(0, 5, n)
label = np.where(score > np.percentile(score, 55), 'Good', 'Bad')

data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'LoanAmount': loan_amount,
    'CreditCardDebt': credit_card_debt,
    'PaymentHistory': payment_history,
    'NumLoans': num_loans,
    'CreditUtilization': (credit_util*100).round(2),
    'EmploymentStatus': employment_status,
    'ExistingDebts': existing_debts,
    'Savings': savings,
    'ScoreApprox': score.round(2),
    'CreditLabel': label
})

# Preprocessing & feature engineering
le = LabelEncoder()
data['CreditLabelEnc'] = le.fit_transform(data['CreditLabel'])
df = data.copy()
df = pd.get_dummies(df, columns=['PaymentHistory','EmploymentStatus'], drop_first=True)
df.drop(columns=['ScoreApprox','CreditLabel'], inplace=True)
# Debt-to-income
df['DebtToIncome'] = (df['ExistingDebts'] + df['LoanAmount'] + df['CreditCardDebt']) / (df['Income'] + 1)
# Financial Stability
df['FinancialStability'] = (df['Savings']/1000)*0.4 + (df['Income']/1000)*0.4 - (df['NumLoans']*2)
# Payment consistency
if 'PaymentHistory_Good' in df.columns:
    df['PaymentConsistency'] = df.get('PaymentHistory_Excellent',0)*1.0 + df.get('PaymentHistory_Good',0)*0.8 + df.get('PaymentHistory_Late',0)*0.4 + df.get('PaymentHistory_Default',0)*0.1
else:
    df['PaymentConsistency'] = 1.0

X = df.drop(columns=['CreditLabelEnc'])
y = df['CreditLabelEnc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Train models
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate
models = {'LogisticRegression': lr, 'DecisionTree': dt, 'RandomForest': rf}
results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    results.append((name, roc))

results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
best_name = results_sorted[0][0]
best_model = models[best_name]

artifact = {'model': best_model, 'scaler': scaler, 'label_encoder': le, 'feature_columns': X.columns.tolist()}

os.makedirs('artifacts', exist_ok=True)
joblib.dump(artifact, 'artifacts/model.pkl')
print('Saved artifacts/model.pkl with model:', best_name)
