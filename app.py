import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Title
st.title("💻 Laptop Price Predictor")

# User inputs
ram = st.selectbox("Select RAM (GB)", [4, 8, 16, 32])
storage = st.selectbox("Select Storage (GB)", [256, 512, 1024])
ssd = st.selectbox("SSD (0 = No, 1 = Yes)", [0, 1])

# Button
if st.button("Predict Price"):
    input_data = np.array([[ram, storage, ssd]])
    prediction = model.predict(input_data)

    st.success(f"Estimated Price: ₹ {int(prediction[0])}")