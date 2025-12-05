# app_improved.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("cats_dogs_transfer_learning.h5")
    # fallback to best_model if you saved that
    try:
        with open("class_names.json", "r") as f:
            class_names = json.load(f)
    except Exception:
        # default order if missing (but better to save class_names at training)
        class_names = ["cats", "dogs"]
    return model, class_names

model, class_names = load_model_and_classes()
IMG_SIZE = 224

st.title("🐱🐶 Cats vs Dogs Classifier")
st.write("Upload an image and the model will tell you if it detects a **Cat** or a **Dog**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def prepare_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype("float32")
    arr = preprocess_input(arr)        # IMPORTANT: same preprocessing used during training
    arr = np.expand_dims(arr, axis=0)
    return arr

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    img_array = prepare_image(image)

    preds = model.predict(img_array)   # shape (1,1)
    prob = float(preds[0][0])
    # Model output is sigmoid for class '1' (class_names[1] hopefully)
    # We saved class_names from train_ds.class_names earlier; ensure order matches how training labeled classes
    # If class_names == ['cats','dogs'] then prob ~ probability of 'dog' (1)
    if len(class_names) == 2:
        label_for_one = class_names[1]  # index 1 -> probability = prob
        label_for_zero = class_names[0]
    else:
        label_for_one = "class_1"
        label_for_zero = "class_0"

    if prob >= 0.5:
        label = f"{label_for_one}"
        confidence = prob
    else:
        label = f"{label_for_zero}"
        confidence = 1 - prob

    st.subheader(f"Prediction: {label}")
    st.subheader(f"Confidence: {confidence:.2f}")
    st.write(f"Raw sigmoid output (prob of `{label_for_one}`): {prob:.4f}")