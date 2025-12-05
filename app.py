import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model (cached so it loads only once)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cats_dogs_transfer_learning.h5")
    return model

model = load_model()

IMG_SIZE = 224

st.title(" Cats vs Dogs Classifier")
st.write("Upload an image and the model will tell you if it detects a **Cat** or a **Dog**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def prepare_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = prepare_image(image)
    
    prediction = model.predict(img_array)[0][0]
    label = "🐶 Dog" if prediction > 0.5 else "🐱 Cat"
    confidence = prediction if prediction > 0.5 else (1 - prediction)

    st.subheader(f"Prediction: {label}")
    st.subheader(f"Confidence: {confidence:.2f}")