import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import time
# Load artifact
artifact = joblib.load('artifacts/model.pkl')
model = artifact['model']
scaler = artifact['scaler']
feature_columns = artifact['feature_columns']
st.set_page_config(page_title='Credit Scoring Dashboard', layout='wide', initial_sidebar_state='expanded')
# Custom CSS for gradient background and card styling
st.markdown("""<style>
.reportview-container {background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);} 
section.main {background: rgba(255,255,255,0.03); padding: 2rem; border-radius:12px;}
.stButton>button{background:linear-gradient(90deg,#0072ff,#00c6ff); color:white;}
.stMetric>div{background: rgba(255,255,255,0.04); border-radius:8px; padding:6px}
.css-1dq8tca{background: linear-gradient(90deg,#041526,#054A91);}
</style>""", unsafe_allow_html=True)
st.title('💳 Credit Scoring Prediction')
col1, col2 = st.columns([1,2])
with col1:
    st.header('Applicant Input')
    st.markdown('**Or upload a CSV with multiple applicants (columns: Age, Income, LoanAmount, CreditCardDebt, PaymentHistory, NumLoans, CreditUtilization, EmploymentStatus, ExistingDebts, Savings)**')
    uploaded = st.file_uploader('Upload credit_scoring_prediction CSV', type=['csv'])
    age = st.slider('Age', 18, 80, 35)
    income = st.number_input('Annual Income (USD)', min_value=1000, max_value=1000000, value=50000, step=1000)
    loan_amount = st.number_input('Loan Amount', min_value=0, max_value=500000, value=15000, step=500)
    credit_card_debt = st.number_input('Credit Card Debt', min_value=0, max_value=200000, value=2000, step=100)
    num_loans = st.slider('Number of Active Loans', 0, 10, 1)
    credit_util = st.slider('Credit Utilization (%)', 0.0, 100.0, 20.0)
    employment = st.selectbox('Employment Status', ['Employed','Self-employed','Unemployed','Retired'])
    existing_debts = st.number_input('Existing Debts', min_value=0, max_value=500000, value=5000, step=100)
    savings = st.number_input('Savings Balance', min_value=0, max_value=1000000, value=8000, step=100)
    payment_hist = st.selectbox('Payment History', ['Excellent','Good','Late','Default'])
    submit = st.button('Predict Creditworthiness')
    batch_predict = False
    uploaded_df = None
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            st.write('Uploaded file preview:')
            st.dataframe(uploaded_df.head())
            if st.button('Run batch prediction on uploaded file'):
                batch_predict = True
        except Exception as e:
            st.error('Failed to read uploaded CSV: ' + str(e))
with col2:
    st.header('Prediction & Insights')
    if submit:
        with st.spinner('Analyzing applicant risk...'):
            time.sleep(1.0)
            # Construct feature vector consistent with training
            input_df = pd.DataFrame([dict(
                Age=age, Income=income, LoanAmount=loan_amount, CreditCardDebt=credit_card_debt, NumLoans=num_loans, CreditUtilization=credit_util, ExistingDebts=existing_debts, Savings=savings
            )])
            input_df['PaymentHistory'] = payment_hist
            # Feature engineering
            input_df['DebtToIncome'] = (input_df['ExistingDebts'] + input_df['LoanAmount'] + input_df['CreditCardDebt']) / (input_df['Income'] + 1)
            input_df['FinancialStability'] = (input_df['Savings']/1000)*0.4 + (input_df['Income']/1000)*0.4 - (input_df['NumLoans']*2)
            pc = 1.0 if payment_hist=='Excellent' else 0.8 if payment_hist=='Good' else 0.4 if payment_hist=='Late' else 0.1
            input_df['PaymentConsistency'] = pc
            # One-hot encode payment and employment to match training columns
            # Create dummies for payment and employment categories to match training
            for ph in ['Excellent','Good','Late','Default']:
                input_df[f'PaymentHistory_{ph}'] = 1 if payment_hist==ph else 0
            for es in ['Employed','Self-employed','Unemployed','Retired']:
                input_df[f'EmploymentStatus_{es}'] = 1 if employment==es else 0
            # Ensure all model feature columns exist
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            # Scale numeric features using the exact feature names the scaler was fitted on
            try:
                scaler_cols = list(getattr(scaler, 'feature_names_in_', []))
                if scaler_cols:
                    for c in scaler_cols:
                        if c not in input_df.columns:
                            input_df[c] = 0
                    # transform in the same column order
                    input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])
                else:
                    num_cols = [c for c in input_df.columns if np.issubdtype(input_df[c].dtype, np.number)]
                    input_df[num_cols] = scaler.transform(input_df[num_cols])
            except Exception:
                num_cols = [c for c in input_df.columns if np.issubdtype(input_df[c].dtype, np.number)]
                input_df[num_cols] = scaler.transform(input_df[num_cols])
            proba = model.predict_proba(input_df[feature_columns])[:,1][0] if hasattr(model, 'predict_proba') else model.predict(input_df[feature_columns])[0]
            pred = 'Good' if proba>0.5 else 'Bad'
            # Credit meter (plotly gauge)
            try:
                import plotly.graph_objects as go
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=proba*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Credit Confidence (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#00cc96" if proba>0.5 else "#ff6b6b"},
                        'steps': [
                            {'range': [0, 50], 'color': '#ff6b6b'},
                            {'range': [50, 80], 'color': '#ffd166'},
                            {'range': [80, 100], 'color': '#00cc96'},
                        ]
                    }
                ))
                st.plotly_chart(gauge, use_container_width=True)
            except Exception:
                st.metric('Prediction', pred, delta=f'Confidence: {proba*100:.1f}%')
            # Risk card
            if pred=='Good':
                st.success(f'Applicant is likely CREDITWORTHY ({proba*100:.1f}% confidence)')
            else:
                st.error(f'Applicant is at HIGH RISK ({proba*100:.1f}% confidence)')
            # Recommendations
            if proba>0.8:
                st.info('Recommendation: Approve loan with standard terms.')
            elif proba>0.5:
                st.info('Recommendation: Consider approval with safeguards (co-signer, collateral).')
            else:
                st.info('Recommendation: Decline or request additional guarantees.')
            # Downloadable report
            if st.checkbox('Generate PDF-like report (text)'):
                report = f"Applicant Prediction: {pred}\nConfidence: {proba*100:.2f}%\nDebt-to-Income: {((existing_debts+loan_amount+credit_card_debt)/ (income+1)):.2f}\n"
                st.download_button('Download Report', report, file_name='credit_report.txt')
            # Download prediction CSV
            try:
                result_df = input_df.copy()
                # inverse transform scaling for readable output if scaler exists
                num_cols = [c for c in result_df.columns if np.issubdtype(result_df[c].dtype, np.number)]
                # create a readable copy (unscaled)
                readable = result_df.copy()
                # add prediction fields
                readable['Prediction'] = pred
                readable['Confidence'] = float(proba)
                csv = readable.to_csv(index=False)
                st.download_button('Download prediction CSV', csv, file_name='prediction.csv', mime='text/csv')
            except Exception:
                pass
            # SHAP local explanation (optional)
            try:
                import shap
                import matplotlib.pyplot as plt
                st.subheader('Local SHAP Feature Importance')
                # build a simple background (zeros) if no background available
                background = np.zeros((1, len(feature_columns)))
                # KernelExplainer can work with any model but may be slow; use small nsamples
                explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:,1], background)
                shap_values = explainer.shap_values(input_df[feature_columns], nsamples=50)
                # shap_values may be nested; convert to array
                vals = np.array(shap_values)
                # for binary output shap returns list-like per class; take mean abs
                if vals.ndim > 2:
                    vals = vals[0]
                vals = np.abs(vals).mean(axis=0)
                feat_imp = pd.Series(vals, index=feature_columns).sort_values(ascending=True)[-10:]
                fig, ax = plt.subplots(figsize=(6,4))
                feat_imp.plot.barh(ax=ax, color='#2b8cbe')
                ax.set_title('Top SHAP feature contributions')
                st.pyplot(fig)
            except Exception as e:
                st.info('SHAP explanation unavailable: ' + str(e))
            # small progress animation to simulate performace checks
            with st.expander('Run quick performance checks'):
                prog = st.progress(0)
                for i in range(5):
                    prog.progress((i+1)*20)
                    time.sleep(0.05)
    # Batch prediction flow
    if batch_predict and uploaded_df is not None:
        st.info('Running preprocessing and batch predictions...')
        dfu = uploaded_df.copy()
        # Ensure required columns exist
        expected_cols = ['Age','Income','LoanAmount','CreditCardDebt','PaymentHistory','NumLoans','CreditUtilization','EmploymentStatus','ExistingDebts','Savings']
        missing = [c for c in expected_cols if c not in dfu.columns]
        if missing:
            st.error(f'Uploaded CSV is missing columns: {missing}')
        else:
            # Create dummies
            for ph in ['Excellent','Good','Late','Default']:
                dfu[f'PaymentHistory_{ph}'] = (dfu['PaymentHistory']==ph).astype(int)
            for es in ['Employed','Self-employed','Unemployed','Retired']:
                dfu[f'EmploymentStatus_{es}'] = (dfu['EmploymentStatus']==es).astype(int)
            # Feature engineering
            dfu['DebtToIncome'] = (dfu['ExistingDebts'] + dfu['LoanAmount'] + dfu['CreditCardDebt']) / (dfu['Income'] + 1)
            dfu['FinancialStability'] = (dfu['Savings']/1000)*0.4 + (dfu['Income']/1000)*0.4 - (dfu['NumLoans']*2)
            dfu['PaymentConsistency'] = dfu.get('PaymentHistory_Excellent',0)*1.0 + dfu.get('PaymentHistory_Good',0)*0.8 + dfu.get('PaymentHistory_Late',0)*0.4 + dfu.get('PaymentHistory_Default',0)*0.1
            # Align to model features
            for col in feature_columns:
                if col not in dfu.columns:
                    dfu[col] = 0
            X_batch = dfu[feature_columns].copy()
            # Scale using scaler's fitted feature names to match training
            try:
                scaler_cols = list(getattr(scaler, 'feature_names_in_', []))
                if scaler_cols:
                    for c in scaler_cols:
                        if c not in X_batch.columns:
                            X_batch[c] = 0
                    X_batch[scaler_cols] = scaler.transform(X_batch[scaler_cols])
                else:
                    num_cols = X_batch.select_dtypes(include=[np.number]).columns.tolist()
                    X_batch[num_cols] = scaler.transform(X_batch[num_cols])
            except Exception:
                num_cols = X_batch.select_dtypes(include=[np.number]).columns.tolist()
                X_batch[num_cols] = scaler.transform(X_batch[num_cols].astype(float))
            # Predict
            probs = model.predict_proba(X_batch)[:,1] if hasattr(model, 'predict_proba') else model.predict(X_batch)
            preds = np.where(probs>0.5, 'Good', 'Bad')
            dfu['Prediction'] = preds
            dfu['Confidence'] = probs
            st.success('Batch prediction completed')
            st.dataframe(dfu.head(50))
            # Risk pie chart
            fig_pie = px.pie(dfu, names='Prediction', title='Risk Distribution')
            st.plotly_chart(fig_pie, use_container_width=True)
            csv = dfu.to_csv(index=False)
            st.download_button('Download batch predictions CSV', csv, file_name='batch_predictions.csv', mime='text/csv')
    else:
        st.info('Fill the form and click Predict to see results.')
