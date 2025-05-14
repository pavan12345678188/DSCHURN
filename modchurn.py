import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import joblib  # type: ignore

# Load model and dataset
model = joblib.load('logistic_regression_model.pkl')
df = joblib.load('df.pkl')

# Set page config
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction App")

# Mapping dictionaries (as used in training)
gender_map = {'Male': 1, 'Female': 0}
partner_map = {'Yes': 1, 'No': 0}
dependents_map = {'Yes': 1, 'No': 0}
phoneservice_map = {'Yes': 1, 'No': 0}
multiplelines_map = {'Yes': 1, 'No': 0, 'No phone service': -1}
internetservice_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
onlinesecurity_map = {'Yes': 1, 'No': 0, 'No internet service': -1}
onlinebackup_map = {'Yes': 1, 'No': 0, 'No internet service': -1}
deviceprotection_map = {'Yes': 1, 'No': 0, 'No internet service': -1}
techsupport_map = {'Yes': 1, 'No': 0, 'No internet service': -1}
streamingtv_map = {'Yes': 1, 'No': 0, 'No internet service': -1}
streamingmovies_map = {'Yes': 1, 'No': 0, 'No internet service': -1}
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
paperlessbilling_map = {'Yes': 1, 'No': 0}
paymentmethod_map = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}

# Average values for default inputs
MonthlyCharges_mean = df['MonthlyCharges'].mean()
TotalCharges_mean = df['TotalCharges'].mean()

# Streamlit form to collect inputs
with st.form("prediction_form"):
    gender = st.selectbox("Gender", options=gender_map.keys())
    SeniorCitizen = st.selectbox("Senior Citizen", options=[0, 1])
    Partner = st.selectbox("Partner", options=partner_map.keys())
    Dependents = st.selectbox("Dependents", options=dependents_map.keys())
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=24)
    PhoneService = st.selectbox("Phone Service", options=phoneservice_map.keys())
    MultipleLines = st.selectbox("Multiple Lines", options=multiplelines_map.keys())
    InternetService = st.selectbox("Internet Service", options=internetservice_map.keys())
    OnlineSecurity = st.selectbox("Online Security", options=onlinesecurity_map.keys())
    OnlineBackup = st.selectbox("Online Backup", options=onlinebackup_map.keys())
    DeviceProtection = st.selectbox("Device Protection", options=deviceprotection_map.keys())
    TechSupport = st.selectbox("Tech Support", options=techsupport_map.keys())
    StreamingTV = st.selectbox("Streaming TV", options=streamingtv_map.keys())
    StreamingMovies = st.selectbox("Streaming Movies", options=streamingmovies_map.keys())
    Contract = st.selectbox("Contract", options=contract_map.keys())
    PaperlessBilling = st.selectbox("Paperless Billing", options=paperlessbilling_map.keys())
    PaymentMethod = st.selectbox("Payment Method", options=paymentmethod_map.keys())
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=MonthlyCharges_mean)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=TotalCharges_mean)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Create input dictionary
    input_data = {
        'gender': gender_map[gender],
        'SeniorCitizen': SeniorCitizen,
        'Partner': partner_map[Partner],
        'Dependents': dependents_map[Dependents],
        'tenure': tenure,
        'PhoneService': phoneservice_map[PhoneService],
        'MultipleLines': multiplelines_map[MultipleLines],
        'InternetService': internetservice_map[InternetService],
        'OnlineSecurity': onlinesecurity_map[OnlineSecurity],
        'OnlineBackup': onlinebackup_map[OnlineBackup],
        'DeviceProtection': deviceprotection_map[DeviceProtection],
        'TechSupport': techsupport_map[TechSupport],
        'StreamingTV': streamingtv_map[StreamingTV],
        'StreamingMovies': streamingmovies_map[StreamingMovies],
        'Contract': contract_map[Contract],
        'PaperlessBilling': paperlessbilling_map[PaperlessBilling],
        'PaymentMethod': paymentmethod_map[PaymentMethod],
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"❌ This customer is likely to churn (probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ This customer is not likely to churn (probability: {prediction_proba:.2f})")
