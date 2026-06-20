import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="kishoresowdi/predictive-model", filename="best_predictive_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Purchase Prediction
st.title("Customer Wellness Tourism Package Purchase Prediction App")
st.write("The Customer Wellness Tourism Package Purchase Prediction App is a tool for \"Visit with Us\" company that predicts whether customers will purchase a wellness package or not.")

st.write("Kindly enter the customer details to check whether they are likely to purchase a wellness tourism package.")

# Collect user input for tourism dataset features
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
CityTier = st.selectbox("City Tier (1, 2, or 3)", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=10)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, value=3)
PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5])
NumberOfTrips = st.number_input("Number of Trips (annually)", min_value=0, value=2)
Passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=25000.0)
TypeofContact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
Occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
Gender = st.selectbox("Gender", ['Male', 'Female'])
ProductPitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
MaritalStatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
Designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[:, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)[0]
    result = "purchase" if prediction == 1 else "not purchase"
    st.write(f"Based on the information provided, the customer is likely to **{result}** the wellness tourism package.")
