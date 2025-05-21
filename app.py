import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('model.h5')

st.title("🧠 Handwritten Digit Classifier")
st.write("Upload a clear image of a handwritten digit (0–9)")

def preprocess_image(image):
    image = image.convert("L")  # grayscale
    img_array = np.array(image)
    
    # Invert if background is light
    if np.mean(img_array) > 127:
        image = ImageOps.invert(image)
        img_array = np.array(image)
    
    # Crop to bounding box of digit (to center)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    
    # Resize to 28x28, keeping aspect ratio and padding
    image = image.resize((20, 20))
    new_img = Image.new('L', (28, 28), (0))  # black background
    new_img.paste(image, ((28 - 20) // 2, (28 - 20) // 2))
    
    # Normalize
    img_array = np.array(new_img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array, new_img

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=150)
    
    processed_img, preview_img = preprocess_image(image)
    st.image(preview_img, caption="Preprocessed Image", width=150)
    
    if st.button("Predict"):
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        st.write(f"### Predicted Digit: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")
