
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

st.title("Classroom Classification AI Web App")
st.write("Upload a classroom image to detect and classify objects.")

# Load your trained TensorFlow model
model = tf.keras.models.load_model("models/my_custom_cnn.h5")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file:
    # Open image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image for your model
    img_array = np.array(image.resize((224, 224)))  # adjust size to your model input
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Run detection
    predictions = model.predict(img_array)
    
    # For demonstration, just show raw predictions
    st.write("Model predictions:", predictions)
    
    # TODO: Add code to draw bounding boxes if your model outputs them