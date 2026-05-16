# Credit Scoring Prediction System

Project overview: End-to-end credit scoring model + Streamlit web app for internship demo.

Installation:
1. Create virtual environment: `python -m venv venv`
2. Activate venv and install deps: `pip install -r requirements.txt`
3. Run the notebook or execute `streamlit run app.py` after running the notebook to produce `artifacts/model.pkl`.

How to run:
- Run the notebook cell-by-cell to generate `artifacts/model.pkl`.
- Then run: `streamlit run app.py`

CSV upload:
- The Streamlit app supports uploading a batch CSV containing applicants with columns: `Age`, `Income`, `LoanAmount`, `CreditCardDebt`, `PaymentHistory`, `NumLoans`, `CreditUtilization`, `EmploymentStatus`, `ExistingDebts`, `Savings`.
- After upload, click "Run batch prediction on uploaded file" to preprocess, predict, visualize, and download results.

Screenshots:
- (Add screenshots here after running the app)

This notebook includes data generation, EDA, feature engineering, model training, evaluation, SHAP (optional), and a professional Streamlit app with a banking-style interface.
