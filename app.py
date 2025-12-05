# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "cats_dogs_transfer_learning.h5"
IMG_SIZE = 224

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title(" Cats vs Dogs Classifier")
st.write("Upload an image and the model will tell you if it detects a **Cat** or a **Dog**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def prepare_image_for_model(pil_img):
    # ensure RGB, resize, convert to array, preprocess for MobileNetV2
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)                # important: use same preprocess_input as training
    arr = np.expand_dims(arr, axis=0)
    return arr

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # updated param

    # Prepare & predict
    img_array = prepare_image_for_model(image)
    pred = model.predict(img_array)[0][0]

    label = "🐶 Dog" if pred > 0.5 else "🐱 Cat"
    confidence = float(pred if pred > 0.5 else (1 - pred))

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.3f}")