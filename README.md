# Cats vs Dogs Image Classification (Transfer Learning + Streamlit)

This project is a deep learning model that classifies uploaded images as either Cat or Dog.
The model is built using TensorFlow (MobileNetV2) and deployed with Streamlit on Render.


**Features**
```
✔️ Upload any image (JPG, JPEG, PNG)

✔️ Model predicts Cat or Dog

✔️ Shows confidence score

✔️ Built using Transfer Learning (MobileNetV2)

✔️ Designed for deployment on Render

✔️ Lightweight and fast inference

```
---

**Model Overview**

The model uses MobileNetV2 pretrained on ImageNet, with:

Frozen early layers

Last layers fine-tuned

Custom dense layers added

Sigmoid output for binary classification


Training process includes:

Automatic train/validation split (80/20)

Normalization + preprocessing

Early stopping

100-epoch fine-tuning



---

**Dataset Structure**

Your dataset folder must look like this:
```
dataset/

   cats/

   dogs/
```
Each folder contains its respective images.

No validation folder needed — the script auto-splits.


---

 **Training Script**

Training script includes:
```
Dataset loading

Preprocessing

Transfer learning setup

Fine-tuning

Model saving (cats_dogs_transfer_learning.h5)
```


---

**Streamlit Web App**

The Streamlit app allows users to:
```
Upload an image

Preview it

Run the model

See prediction + confidence score

```
**Example output:**
```
Prediction: 🐱Cat
Confidence: 0.93

or

Prediction: 🐶 Dog
Confidence: 0.88

```
---

**Deploy on Render**

Deployment setup includes:
```
app.py → Streamlit application

requirements.txt → Dependencies

Render detects repository & deploys automatically

Supports auto-redeploy when you push updates
```


---

**Installation (Run Locally)**

Clone the repo:
```
git clone https://github.com/day6ix/cats-dogs-classifier.git
cd cats-dogs-classifier
```
Install dependencies:
```
pip install -r requirements.txt
```
Run app:
```
streamlit run app.py
```

---

**Technologies Used**
```
Python

TensorFlow / Keras

MobileNetV2

Streamlit

NumPy + PIL

Render (cloud hosting)

```

---
```
Future Improvements

Add more images to increase accuracy

Add data augmentation

Train with a larger dataset like Kaggle Dogs vs Cats

Move to 3-class model (Cat / Dog / Other) in future
```

