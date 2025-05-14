import streamlit as st # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore

# Load model and data
model = joblib.load('best_model_churn.pkl')
df = joblib.load('df.pkl')  # contains encoded structure

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict whether they are likely to churn.")

# Create form for user input
with st.form("prediction_form"):
    user_input = {}
    for col in df.drop('Churn', axis=1).columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))
        else:
            user_input[col] = st.selectbox(f"{col}", options=df[col].unique())

    submitted = st.form_submit_button("Predict Churn")

# On submit, make prediction
if submitted:
    input_df = pd.DataFrame([user_input])
    # Ensure columns match
    input_df = input_df[df.drop('Churn', axis=1).columns]
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]  # probability of churn
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"❌ This customer is likely to churn (probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ This customer is not likely to churn (probability: {prediction_proba:.2f})")
