import streamlit as st  # type: ignore
import pandas as pd     # type: ignore
import joblib           # type: ignore

# Load model and preprocessed data
model = joblib.load('best_model_churn.pkl')
df = joblib.load('df.pkl')  # Data used for training

# Streamlit setup
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction App")

# Category mapping dictionaries
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
paymentmethod_map = {
    'Electronic check': 0,
    'Mailed check': 1,
    'Bank transfer (automatic)': 2,
    'Credit card (automatic)': 3
}

# Default values
MonthlyCharges_default = df['MonthlyCharges'].mean()
TotalCharges_default = df['TotalCharges'].mean()

# Input form
with st.form("prediction_form"):
    customerID = st.text_input("Customer ID")
    gender = st.selectbox("Gender", list(gender_map.keys()))
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", list(partner_map.keys()))
    Dependents = st.selectbox("Dependents", list(dependents_map.keys()))
    tenure = st.number_input("Tenure", min_value=0, max_value=100, value=30)
    PhoneService = st.selectbox("Phone Service", list(phoneservice_map.keys()))
    MultipleLines = st.selectbox("Multiple Lines", list(multiplelines_map.keys()))
    InternetService = st.selectbox("Internet Service", list(internetservice_map.keys()))
    OnlineSecurity = st.selectbox("Online Security", list(onlinesecurity_map.keys()))
    OnlineBackup = st.selectbox("Online Backup", list(onlinebackup_map.keys()))
    DeviceProtection = st.selectbox("Device Protection", list(deviceprotection_map.keys()))
    TechSupport = st.selectbox("Tech Support", list(techsupport_map.keys()))
    StreamingTV = st.selectbox("Streaming TV", list(streamingtv_map.keys()))
    StreamingMovies = st.selectbox("Streaming Movies", list(streamingmovies_map.keys()))
    Contract = st.selectbox("Contract", list(contract_map.keys()))
    PaperlessBilling = st.selectbox("Paperless Billing", list(paperlessbilling_map.keys()))
    PaymentMethod = st.selectbox("Payment Method", list(paymentmethod_map.keys()))
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=MonthlyCharges_default)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=TotalCharges_default)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_data = {
        'customerID': customerID,
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

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Match training column order (excluding 'Churn')
    expected_columns = df.drop(columns='Churn').columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # default fallback
    input_df = input_df[expected_columns]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Display result
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"❌ Customer is likely to churn (probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer is not likely to churn (probability: {probability:.2f})")
