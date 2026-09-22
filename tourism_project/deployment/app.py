import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Load the trained model ---
MODEL_PATH = 'tourism_project/deployment/best_model.joblib'

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        st.stop()
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model(MODEL_PATH)

st.title("✈️ Tourism Package Purchase Prediction")
st.markdown("--- Say hello to your next adventure! ---")

st.write("### Enter Customer Details to Predict Purchase")

# --- Input features ---
with st.form("prediction_form"):
    st.header("Personal Information")
    age = st.slider("Age", 18, 70, 30)
    occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    marital_status = st.selectbox("Marital Status", ['Married', 'Divorced', 'Single'])
    monthly_income = st.number_input("Monthly Income (USD)", min_value=1000.0, max_value=100000.0, value=25000.0, step=100.0)
    designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])

    st.header("Travel Preferences & Interaction")
    typeofcontact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.slider("Duration of Pitch (minutes)", 5, 60, 15)
    number_of_person_visiting = st.slider("Number of Persons Visiting", 1, 5, 2)
    number_of_followups = st.slider("Number of Follow-ups", 1, 6, 3)
    product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
    preferred_property_star = st.slider("Preferred Property Star (1-5)", 1, 5, 3)
    number_of_trips = st.slider("Number of Trips (annually)", 1, 15, 3)
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score (1-5)", 1, 5, 3)
    own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    number_of_children_visiting = st.slider("Number of Children Visiting", 0, 3, 0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'Age': [age],
            'TypeofContact': [typeofcontact],
            'CityTier': [city_tier],
            'DurationOfPitch': [duration_of_pitch],
            'Occupation': [occupation],
            'Gender': [gender],
            'NumberOfPersonVisiting': [number_of_person_visiting],
            'NumberOfFollowups': [number_of_followups],
            'ProductPitched': [product_pitched],
            'PreferredPropertyStar': [preferred_property_star],
            'MaritalStatus': [marital_status],
            'NumberOfTrips': [number_of_trips],
            'Passport': [passport],
            'PitchSatisfactionScore': [pitch_satisfaction_score],
            'OwnCar': [own_car],
            'NumberOfChildrenVisiting': [number_of_children_visiting],
            'Designation': [designation],
            'MonthlyIncome': [monthly_income]
        })

        # Make prediction
        try:
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)[:, 1]

            st.write("### Prediction Result")
            if prediction[0] == 1:
                st.success(f"This customer is predicted to **PURCHASE** the tourism package! 🎉 (Confidence: {prediction_proba[0]*100:.2f}%) ")
            else:
                st.info(f"This customer is predicted **NOT TO PURCHASE** the tourism package. (Confidence: {(1-prediction_proba[0])*100:.2f}%) ")

            st.subheader("Input Data for Prediction")
            st.dataframe(input_data)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

st.markdown("--- Developed by your MLOps Engineer team ---")
