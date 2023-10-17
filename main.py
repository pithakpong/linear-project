import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import time
fig = plt.figure()

optimizer = Adam(learning_rate=0.001)
model = tf.keras.models.load_model("emotion_model.h5",compile=False)
model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.optimizers import Adam

# # Load the model with the custom optimizer
# custom_optimizer = CustomAdamOptimizer(learning_rate=0.001, weight_decay=0.001)
# model = keras.models.load_model('emotion_model.h5', custom_objects={'CustomAdamOptimizer': custom_optimizer})

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
st.title(' 🤠 🤠Emotion classifier 🤯 🤯')

def main():
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        # Display the resized image
        st.image(img_array, caption="Uploaded Image", use_column_width=True)
        class_btn = st.button("Classify")
        if class_btn:
            if uploaded_file is None:
                st.write("Error!!, please upload an image")
            else:
                with st.spinner('Model working....'):
                    plt.imshow(image)
                    plt.axis("off")
                    predictions = predict(img_array)
                    # time.sleep(1)
                    st.success('Classified')
                    st.write(predictions)
                    # st.pyplot(fig)

def predict(img_array):
    height, width, _ = img_array.shape
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized_image = cv2.resize(gray_image, (48, 48))
    resized_image = resized_image / 255.0
    img = resized_image.reshape(1, 48, 48, 1)
    count = 0

    # Make predictions using the loaded model
    predictions = model.predict(img)
    
    # Define emotion labels
    emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # Display the predicted emotion
    predicted_emotion = emotion_labels[np.argmax(predictions)]
    time.sleep(1)
    st.latex(f'Predicted Emotion: ***** {predicted_emotion} *****')
    for i in range(len(predictions)):
        predictions[i] = predictions[i] * 100
        
    # chart_data = st.dataframe(
    #     {
    #    "Angry": predictions[0],
    #    "Fear": predictions[1],
    #    "Happy": predictions[2],
    #    "Sad": predictions[3],
    #    "Surprise": predictions[4],
    #    "Surprise": predictions[5],
    #    }
    # )
    
    st.title("_________________________________")

    for i in np.nditer(predictions):
        st.latex(f"{emotion_labels[count]} : {i}")
        count += 1 

if __name__ == "__main__":
    main()