import streamlit as st
from PIL import Image
import numpy as np
import random

st.set_page_config(page_title="Lung Cancer Detection", layout="centered")
st.title("🫁 Lung Cancer Detection")
st.write("Upload a lung tissue image for classification")

# نموذج وهمي للاختبار
def predict_dummy(image):
    classes = ['Adenocarcinoma', 'Benign', 'Squamous Cell Carcinoma']
    predicted_class = random.choice(classes)
    confidence = random.uniform(70, 98)
    return predicted_class, confidence

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with st.spinner("Analyzing..."):
        predicted_class, confidence = predict_dummy(image)
    
    st.success(f"### 🔬 Prediction: **{predicted_class}**")
    st.info(f"### 📊 Confidence: **{confidence:.2f}%**")
    st.caption("Note: This is a demo version. Full model coming soon.")
