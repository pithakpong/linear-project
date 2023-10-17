import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)
model = tf.keras.models.load_model("emotion_model.h5",compile=False)
model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.optimizers import Adam

# # Load the model with the custom optimizer
# custom_optimizer = CustomAdamOptimizer(learning_rate=0.001, weight_decay=0.001)
# model = keras.models.load_model('emotion_model.h5', custom_objects={'CustomAdamOptimizer': custom_optimizer})


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    height, width, _ = img_array.shape
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized_image = cv2.resize(gray_image, (48, 48))
    resized_image = resized_image / 255.0
    img = resized_image.reshape(1, 48, 48, 1)

    # Display the resized image
    st.image(resized_image, caption="Resized Image", use_column_width=True)

    # Make predictions using the loaded model
    predictions = model.predict(img)

    # Define emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # Display the predicted emotion
    predicted_emotion = emotion_labels[np.argmax(predictions)]
    st.write(f'Predicted Emotion: {predicted_emotion}')
