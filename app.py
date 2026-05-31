import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Salary Predictor",
                   page_icon="💰", layout="centered")
st.title("💰 Employee Salary Predictor (>100k)")
st.write("Enter the employee's details below to predict if their salary exceeds $100,000.")


@st.cache_resource
def load_ml_assets():
    with open('salaries.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('le_1.pkl', 'rb') as f:
        company_encoder = pickle.load(f)
    with open('le_2.pkl', 'rb') as f:
        job_encoder = pickle.load(f)
    with open('le_3.pkl', 'rb') as f:
        degree_encoder = pickle.load(f)

    return model, company_encoder, job_encoder, degree_encoder


try:
    model, company_encoder, job_encoder, degree_encoder = load_ml_assets()
except FileNotFoundError as e:
    st.error(f"📁 Missing file error: {e}")
    st.info("Make sure 'salaries.pkl', 'le_1.pkl', 'le_2.pkl', and 'le_3.pkl' are all saved in this exact folder.")
    st.stop()

st.subheader("Step 1: Enter Employee Details")
selected_company = st.selectbox(
    "Select Company:", options=company_encoder.classes_)
selected_job = st.selectbox("Select Job Role:", options=job_encoder.classes_)
selected_degree = st.selectbox(
    "Select Highest Degree:", options=degree_encoder.classes_)
st.write("---")
st.subheader("Step 2: Model Prediction")

if st.button("Predict Salary Bracket", type="primary"):

    # Transform the selected text back into the exact numbers the model learned
    company_encoded = company_encoder.transform([selected_company])[0]
    job_encoded = job_encoder.transform([selected_job])[0]
    degree_encoded = degree_encoder.transform([selected_degree])[0]

    input_data = np.array([[company_encoded, job_encoded, degree_encoded]])

    # Run the prediction (outputs 0 or 1)
    prediction = model.predict(input_data)

    # 5. Display the output dynamically
    if prediction[0] == 1:
        st.success(
            "🎉 **Prediction:** This employee is likely earning **MORE than $100k**!")
    else:
        st.info(
            "📉 **Prediction:** This employee is likely earning **LESS than $100k**.")

    # Optional: Display probability confidence if available
    try:
        probabilities = model.predict_proba(input_data)
        confidence = max(probabilities[0]) * 100
        st.write(f"*Model Confidence: {confidence:.1f}%*")
    except:
        pass
