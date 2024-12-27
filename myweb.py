import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the saved models
cnn_model = tf.keras.models.load_model('cnn_model1.h5')
vgg16_model = tf.keras.models.load_model('vgg16_model1.h5')

# Class names (Replace with your actual class names)
class_names = ['Early_Blight', 'Healthy', 'Late_Blight']

# General recommendations for diseases
recommendations = {
    "Early_Blight": "Use fungicides and remove any infected plants to prevent the spread.",
    "Late_Blight": "Use a fungicide specifically for late blight and remove any infected plants immediately to prevent further spread.",
    "Healthy": "The plant is healthy. No action needed."
}

# Function to preprocess the image for prediction
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = np.array(img) / 255.0  # Rescale image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Main function
def main():
    st.title("Potato Classification App")
    st.image("header_image.jpg", use_column_width=True)  # Add header image
    st.write("Upload an image of a potato to classify it using both models.")

    # Sidebar content
    st.sidebar.title("About")
    st.sidebar.info(
        "This application classifies potato leaf images into Early Blight, Late Blight, or Healthy categories "
        "using deep learning models (CNN and VGG16). Recommendations are provided based on the prediction."
    )
    st.sidebar.title("Developers")
    st.sidebar.text("Nouman Ali\nSamee Haider")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Columns for side-by-side buttons and results
        col1, col2 = st.columns(2)

        with col1:
            st.write("**CNN Model**")
            if st.button("Check with CNN Model"):
                # Preprocess and predict with CNN model
                img_array = preprocess_image(img)
                predictions_cnn = cnn_model.predict(img_array)
                predicted_class_cnn = np.argmax(predictions_cnn)
                confidence_cnn = predictions_cnn[0][predicted_class_cnn] * 100
                disease_cnn = class_names[predicted_class_cnn]

                # Display CNN results
                st.write(f"Prediction: {class_names[predicted_class_cnn]}")
                st.write(f"Confidence: {confidence_cnn:.2f}%")
                st.write("Recommendations:")
                st.write(recommendations[disease_cnn])

        with col2:
            st.write("**VGG16 Model**")
            if st.button("Check with VGG16 Model"):
                # Preprocess and predict with VGG16 model
                img_array = preprocess_image(img)
                predictions_vgg16 = vgg16_model.predict(img_array)
                predicted_class_vgg16 = np.argmax(predictions_vgg16)
                confidence_vgg16 = predictions_vgg16[0][predicted_class_vgg16] * 100
                disease_vgg16 = class_names[predicted_class_vgg16]

                # Display VGG16 results
                st.write(f"Prediction: {class_names[predicted_class_vgg16]}")
                st.write(f"Confidence: {confidence_vgg16:.2f}%")
                st.write("Recommendations:")
                st.write(recommendations[disease_vgg16])

    # Footer styling and content
    footer = """
    <style>
    .footer {
        font-size: 14px;
        color: #fff;
        text-align: center;
        padding: 5px 10px;
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #333333;
    }
    </style>
    <div class='footer'>
        <a href='https://github.com/noumzz' target='_blank' style='margin-right: 10px; text-decoration: none; font-weight: bold;'>GitHub Link 1</a>
        Developed by <b>Nouman Ali</b> and <b>Samee Haider</b>
        <a href='https://github.com/sameehaider' target='_blank' style='margin-left: 10px; text-decoration: none; font-weight: bold;'>GitHub Link 2</a>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
