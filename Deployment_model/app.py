import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load your trained model
model = joblib.load('house_price_model.pkl')

# Function to encode categorical location feature
def encode_location(location):
    label_encoder = LabelEncoder()
    # Assuming you have already fitted the encoder with your training data
    locations = ['CHENNAI', 'KANYAKUMARI', 'MADURAI','COIMBATORE','NAGAPATTINAM','NAMAKKAL','SALEM','SIVAGANGAI']  # Replace with your actual locations
      # Replace with your actual locations
    label_encoder.fit(locations)
    return label_encoder.transform([location])[0]

# Streamlit UI with Emojis
st.title("🏠 House Price Prediction 🏠")


# Input fields for user with emojis
area = st.number_input("Enter Area in Square Feet 📏", min_value=0, step=1)
bedrooms = st.number_input("Enter Number of Bedrooms 🛏", min_value=0, step=1)
bathrooms = st.number_input("Enter Number of Bathrooms 🚿", min_value=0, step=1)
location = st.selectbox("Select Location 📍", ['CHENNAI', 'KANYAKUMARI', 'MADURAI','COIMBATORE','NAGAPATTINAM','NAMAKKAL','SALEM','SIVAGANGAI'])  # Replace with your actual locations

# Convert location to a numerical value
location_encoded = encode_location(location)

# Prediction button with emoji
if st.button("🔮 Predict House Price"):
    input_data = np.array([area, bedrooms, bathrooms, location_encoded]).reshape(1, -1)
    prediction = model.predict(input_data)
    st.write(f"💰 Predicted House Price: $ {prediction[0]:,.2f} 💰")