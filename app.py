import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("intel_scene_classifier.keras")

# Scene classes
class_names = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

# App title
st.title("Natural Scene Image Classifier")

st.write(
    "Upload a natural scene image to predict its category."
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# Prediction
if uploaded_file is not None:

    img = image.load_img(
        uploaded_file,
        target_size=(150, 150)
    )

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction) * 100

    st.success(
        f"Predicted Scene: {predicted_class}"
    )

    st.write(
        f"Confidence Score: {confidence:.2f}%"
    )
