
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("cats_vs_dogs_model.h5")

# Set up Streamlit interface
st.title("Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog, and the model will predict the class.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for model prediction
    img = img.resize((150, 150))  # Resize to match model's input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale as done in training

    # Make prediction
    prediction = model.predict(img_array)[0][0]  # Get the predicted probability

    # Interpret the prediction
    if prediction > 0.5:
        st.write("Prediction: Dog")
    else:
        st.write("Prediction: Cat")
