import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.layers import Rescaling
import base64

# ===========================
# Function to set background
# ===========================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp .main {{
            background-color: rgba(255, 255, 255, 0.75);  /* semi-transparent panel */
            padding: 20px;
            border-radius: 15px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background (image in the same folder as app.py)
set_background("calssroom.jpg")

# ===========================
# App title
# ===========================
st.title("Classroom Classification AI Web App")

# ===========================
# Load model
# ===========================
model_path = "my_custom_cnn.h5"
model = tf.keras.models.load_model(model_path)
st.success("Model loaded successfully!")

# ===========================
# Class names
# ===========================
class_names = ['Chair', 'Keyboard', 'Monitor', 'Mouse', 'PC', 'Whiteboard ']

# ===========================
# Image preprocessing function
# ===========================
def preprocess_image(image, target_size=(224, 224)):
    img = image.convert('RGB').resize(target_size)
    x = np.array(img, dtype=np.float32)
    # Check if model has a Rescaling layer
    has_rescaling = any(isinstance(l, Rescaling) for l in model.layers)
    if not has_rescaling:
        x = x / 255.0
    return np.expand_dims(x, axis=0)

# ===========================
# Image uploader and prediction
# ===========================
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and predict
    x = preprocess_image(image)
    prediction = model.predict(x)
    pred_index = np.argmax(prediction[0])
    pred_name = class_names[pred_index]

    # Display raw probabilities
    st.write("Raw prediction probabilities:")
    for i, name in enumerate(class_names):
        st.write(f"{name}: {prediction[0][i]*100:.2f}%")

    # Display predicted class
    st.success(f"Predicted class: {pred_name}")
