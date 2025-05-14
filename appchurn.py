import streamlit as st # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore

# Load model and data
model = joblib.load('best_model_churn.pkl')
df = joblib.load('df.pkl')  # Preprocessed DataFrame

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction App")

# --- Category Mapping Dictionaries ---
gender_map = {'Male': 1, 'Female': 0}
partner_map = {'Yes': 1, 'No': 0}
dependents_map = {'Yes': 1, 'No': 0}
phoneservice_map = {'Yes': 1, 'No': 0}
internetservice_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
payment_map = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}

# Collect input from user
with st.form("prediction_form"):
    gender = st.selectbox("Gender", options=list(gender_map.keys()))
    SeniorCitizen = st.number_input("SeniorCitizen", min_value=0.0, max_value=1.0, value=0.0)
    Partner = st.selectbox("Partner", options=list(partner_map.keys()))
    Dependents = st.selectbox("Dependents", options=list(dependents_map.keys()))
    tenure = st.number_input("Tenure", min_value=0.0, max_value=100.0, value=30.0)
    PhoneService = st.selectbox("PhoneService", options=list(phoneservice_map.keys()))
    # Add more fields as needed...

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Convert inputs to numeric values as per training data
    input_data = {
        'gender': gender_map[gender],
        'SeniorCitizen': SeniorCitizen,
        'Partner': partner_map[Partner],
        'Dependents': dependents_map[Dependents],
        'tenure': tenure,
        'PhoneService': phoneservice_map[PhoneService],
        # Add more mappings here...
    }

    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    input_df = input_df[df.drop('Churn', axis=1).columns]  # Ensure correct column order

    # Predict
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    # Show result
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"❌ This customer is likely to churn (probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ This customer is not likely to churn (probability: {prediction_proba:.2f})")
