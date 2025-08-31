import streamlit as st
import tensorflow as tf
import os
import gdown
from PIL import Image
import numpy as np
from tensorflow.keras.layers import Rescaling

# ===========================
# Foggy / translucent background
# ===========================
def set_foggy_background(color="white", opacity=0.6):
    st.markdown(
        f"""
        <style>
        /* app background color */
        .stApp {{
            background-color: {color};
        }}
        /* main panel foggy effect */
        .stApp .main {{
            background-color: rgba(255, 255, 255, {opacity});
            padding: 20px;
            border-radius: 15px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_foggy_background(color="white", opacity=0.6)

# ===========================
# App title
# ===========================
st.title("Classroom Classification AI Web App")

# ===========================
# Models info
# ===========================
models_info = {
    "My Custom CNN": {
        "filename": "models/my_custom_cnn.h5",
        "gdrive_url": "https://drive.google.com/uc?id=1OzRiSRs-k0L8B1dro4JyW5rjPv7Zzlnp"
    },
    "CNN VGG16": {
        "filename": "models/cnn_modelVGG16.h5",
        "gdrive_url": "https://drive.google.com/uc?id=1nb_4h9nOzpUo9oxEvNujEL5Yw02ptB9X"
    }
}

# ===========================
# Create models folder
# ===========================
os.makedirs("models", exist_ok=True)

# ===========================
# Load models
# ===========================
loaded_models = {}
for name, info in models_info.items():
    if not os.path.exists(info["filename"]):
        st.info(f"Downloading {name} model, please wait...")
        gdown.download(info["gdrive_url"], info["filename"], quiet=False)
        st.success(f"{name} model downloaded!")
    loaded_models[name] = tf.keras.models.load_model(info["filename"])
    st.success(f"{name} model loaded successfully!")

# ===========================
# Sidebar: choose model
# ===========================
selected_model_name = st.sidebar.selectbox("Choose a model", list(loaded_models.keys()))
model_to_use = loaded_models[selected_model_name]

# ===========================
# Class names (adjust if different per model)
# ===========================
class_names = ['Chair', 'Keyboard', 'Monitor', 'Mouse', 'PC', 'Whiteboard ']

# ===========================
# Image uploader
# ===========================
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # ===========================
    # Preprocess image
    # ===========================
    img_array = np.array(image.resize((224, 224)), dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # ===========================
    # Run prediction
    # ===========================
    prediction = model_to_use.predict(img_array)

    # ===========================
    # Display raw probabilities
    # ===========================
    st.write("Raw prediction probabilities:")
    for i, name in enumerate(class_names):
        st.write(f"{name}: {prediction[0][i]*100:.2f}%")

    # ===========================
    # Display predicted class
    # ===========================
    pred_index = np.argmax(prediction[0])
    pred_name = class_names[pred_index]
    st.write("Predicted class:", pred_name)
