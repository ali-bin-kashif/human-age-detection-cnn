import streamlit as st
import tensorflow as tf
from mtcnn import MTCNN
import numpy as np
from PIL import Image
import cv2

# Load the saved TensorFlow model
model = tf.keras.models.load_model('saved_models/my_model.h5')

if model:
    print("Model loaded successfully.")
else:
    print("Failed to load the model.")

# Define the age classes based on your training dataset
AGE_CLASSES = ['MIDDLE', 'OLD', 'YOUNG']

# Initialize MTCNN face detector
detector = MTCNN()

# Function to preprocess the face for the model
def preprocess_face(face):
    # face = cv2.resize(face, (48, 48))  # Resize to model input size
    # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if required
    # face_array = np.array(face) / 255.0  # Normalize pixel values
    # face_array = np.expand_dims(face_array, axis=-1)  # Add channel dimension
    face_array = np.expand_dims(face, axis=0)  # Add batch dimension
    print(face_array.shape)
    return face_array

# Streamlit app
st.title("Age Detection App with Face Detection")
st.write("Capture an image with your camera to detect faces and estimate age.")

# Capture image from the camera
captured_image = st.camera_input("Take a picture")

if captured_image is not None:
    # Load image
    image = Image.open(captured_image)
    image_np = np.array(image)

    # Detect faces using MTCNN
    faces = detector.detect_faces(image_np)

    if faces:
        # st.image(image, caption="Original Image", use_column_width=True)
        
        for i, face in enumerate(faces):
            x, y, width, height = face['box']
            face_region = image_np[y:y+height, x:x+width]

            # Preprocess face and predict age
            preprocessed_face = preprocess_face(face_region)
            prediction = model.predict(preprocessed_face)
            predicted_age_class = AGE_CLASSES[np.argmax(prediction)]
            print(predicted_age_class)

            # Draw bounding box and label on the image
            cv2.rectangle(image_np, (x, y), (x + width, y + height), (0, 255, 0), 2)
            # cv2.putText(image_np, predicted_age_class, (x, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Display the processed image with predictions
        st.image(image_np, caption="Detected Faces with Age Prediction", use_column_width=True)
        st.write("Predicted Age: ", predicted_age_class)
    else:
        st.write("No faces detected in the image.")
