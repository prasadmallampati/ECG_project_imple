import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to the required input shape
    resized_image = image.resize((224, 224))
    # Convert the image to a numpy array and normalize pixel values
    processed_image = np.array(resized_image) / 255.0
    # Expand dimensions to create a batch for prediction
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('keras_model.h5')  # Replace 'your_model.h5' with the path to your model file
    return model

# Define class labels
class_labels = ['Abnormal Heartbeat Patients', 'Myocardial Infarction Patients', 'Normal Person', 'Patient with History of Myocardial Infraction']

# Load the model
model = load_model()

# Main Streamlit app
st.title('Cardiovascular Disease Detection in ECG Images')
st.write('Upload an ECG image for disease detection')

# Upload image
uploaded_file = st.file_uploader("Choose an ECG image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded ECG Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display the prediction result
    st.write(f'Prediction: {class_labels[predicted_class]}')
    st.write(f'Confidence: {confidence * 100:.2f}%')
