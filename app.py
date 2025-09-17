import streamlit as st
import tensorflow as tf
import os
import gdown
from PIL import Image
import numpy as np
import base64

# ===========================
# Background image function
# ===========================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp .main {{
            background-color: rgba(255, 255, 255, 0.6);  /* semi-transparent overlay */
            padding: 20px;
            border-radius: 15px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Add background (make sure file is in the same folder as app.py)
set_background("Untitled design.png")

# ===========================
# App title
# ===========================
st.title("Classroom Classification AI Web App")

# ===========================
# Models info (all your models)
# ===========================
models_info = {
    "ResNet50": {
        "filename": "models/capResNet50.h5",
        "gdrive_url": "https://drive.google.com/uc?id=1zvtYbo92sjAgtgNzfbWaTiyAwM-VWBEW"
    },
    "ResNet101": {
        "filename": "models/capResNet101.h5",
        "gdrive_url": "https://drive.google.com/uc?id=1inqcxDkiiBBQAp_XIQqHNd_gb-3fmHT-"
    },
    "VGG16": {
        "filename": "models/capVGG16.h5",
        "gdrive_url": "https://drive.google.com/uc?id=1KIj20nQ68_chvNFIm2PWXzYh2GIIAJRe"
    },
    "VGG19": {
        "filename": "models/capVGG19.h5",
        "gdrive_url": "https://drive.google.com/uc?id=1ygMe8Iv92r338vImfSB2kiWgpvJyInqS"
    },
    "CustomCNN_3": {
        "filename": "models/customCNN_3.h5",
        "gdrive_url": "https://drive.google.com/uc?id=1Y77wGofPP2Lx89ZW9d8Vrv33nEsAHeNU"
    },
    "CustomCNN_5": {
        "filename": "models/customCNN_5.h5",
        "gdrive_url": "https://drive.google.com/uc?id=1J4rtV7yMpiv5ZTmca4MWLV5IdKHZNmjp"
    },
    "CustomCNN_7": {
        "filename": "models/customCNN_7.h5",
        "gdrive_url": "https://drive.google.com/uc?id=1UZ36ow9p1OycgV_zXsZpRjdcdG61pCTs"
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
# Updated class names
# ===========================
class_names = ['Bag', 'Chair', 'Keyboard', 'Mobile', 'Monitor', 'Mouse', 'PC', 'Whitebord']

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
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1,224,224,3)

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
