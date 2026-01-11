#########################################################
########### Web App of CNN Classification ###############
#########################################################

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------
# Page Config
st.set_page_config(page_title="Smart Store Product Classification", layout="centered")

# -------------------------------
# Load trained model (cached)
@st.cache_resource
def load_cnn_model():
    return load_model("CNN_model.keras")

model = load_cnn_model()

# -------------------------------
# Class labels
class_labels = [
    'cake', 'candy', 'cereal', 'chips', 'chocolate', 'coffee',
    'fish', 'honey', 'jam', 'milk', 'oil', 'pasta', 'rice',
    'soda', 'sugar', 'tea', 'vinegar', 'water'
]

# -------------------------------
# UI
st.title("🛒 Smart Store Product Classification System")
st.write("Upload an image and the CNN model will classify the product.")

uploaded_file = st.file_uploader("Select the image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((244, 244))   # Change if your model expects a different input size
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])

    # Show Results
    st.success(f"✅ Predicted Label: {predicted_label}")
    st.info(f"📌 Confidence: {confidence:.2f}")
