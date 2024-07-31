import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image


# Load the saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Models\covid_classification_model.h5")


model = load_model()

# Define class names
class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

st.title("Medical X-ray Image Classifier")

uploaded_file = st.file_uploader("Choose an X-ray image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

    # Display bar chart of probabilities
    st.bar_chart(dict(zip(class_names, prediction[0])))

st.write("Note: This app is not yet ready for actual medical diagnosis.")
