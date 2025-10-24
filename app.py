# Importing Necessary Libraries
import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results

# Loading the Model and saving to cache
# Update deprecated st.cache to st.cache_resource
@st.cache_resource
def load_model(path):
    # Xception Model
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # DenseNet Model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)
    outputs = tf.keras.layers.average([densenet_output, xception_output])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Loading the Weights of Model
    model.load_weights(path)
    
    return model

# Removing Menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Loading the Model
model = load_model('model.h5')

# Set page configuration with dark green background
st.markdown("""
    <style>
    .stApp {
        background-color: #1e4d40;
    }
    .css-18e3th9 {
        padding: 1rem 2rem;
    }
    .uploadedImage {
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Main header section - with consistent border radius
st.markdown("""
    <div style='background-color: #2a8c5e; padding: 2rem; border-radius: 15px; margin-bottom: 1.5rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);'>
    <h1 style='text-align: center; color: white; margin-bottom: 1rem;
    font-weight: 600; letter-spacing: 0.5px;'>
    🌿 AI-Powered Betel Leaf Disease Detection
    </h1>
    <p style='text-align: center; color: white; font-size: 1rem;
    max-width: 800px; margin: 0 auto;'>
    Upload a clear image of betel leaves to detect common diseases with our advanced AI model
    </p>
    </div>
    """, unsafe_allow_html=True)

# Disease category cards - all with consistent styling
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div style='background: #fff8ee; padding: 1.5rem; border-radius: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 100%;'>
        <h3 style='color: #8B4513; text-align: center; font-size: 1.5rem; 
        margin-bottom: 1rem; font-weight: 600;'>Anthracnose</h3>
        <p style='color: #555; font-size: 1rem; text-align: center;'>
        🔍 Dark spots with yellow halos
        </p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style='background: #f0f8ff; padding: 1.5rem; border-radius: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 100%;'>
        <h3 style='color: #0066cc; text-align: center; font-size: 1.5rem; 
        margin-bottom: 1rem; font-weight: 600;'>Blight</h3>
        <p style='color: #555; font-size: 1rem; text-align: center;'>
        🔍 Brown patches on leaves
        </p>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div style='background: #f0fff0; padding: 1.5rem; border-radius: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 100%;'>
        <h3 style='color: #2e8b57; text-align: center; font-size: 1.5rem; 
        margin-bottom: 1rem; font-weight: 600;'>Healthy</h3>
        <p style='color: #555; font-size: 1rem; text-align: center;'>
        🔍 Normal green leaves
        </p>
        </div>
        """, unsafe_allow_html=True)

# Upload section - with consistent border radius
st.markdown("""
    <div style='background: white; padding: 2rem; border-radius: 15px;
    margin: 2rem 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center;'>
    <h2 style='color: #2a8c5e; margin-bottom: 1rem;'>📤 Upload Leaf Image</h2>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["png", "jpg"], 
                               help="Select a clear image of betel leaves",
                               label_visibility="collapsed")

st.markdown("</div>", unsafe_allow_html=True)

# Initialize result variable
result = None

# If there is a uploaded file, start making prediction
if uploaded_file is not None:
    # Display progress and text
    progress = st.text("Analyzing image...")
    my_bar = st.progress(0)
    i = 0

    # Reading the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    
    # Display the image with proper resampling
    st.image(np.array(Image.fromarray(
        np.array(image)).resize((700, 400), Image.Resampling.LANCZOS)), width=None)
    my_bar.progress(i + 40)

    # Cleaning the image
    image = clean_image(image)

    # Making the predictions
    predictions, predictions_arr = get_prediction(model, image)
    my_bar.progress(i + 30)

    # Making the results
    result = make_results(predictions, predictions_arr)

    # Removing progress bar and text after prediction done
    my_bar.progress(i + 30)
    progress.empty()
    i = 0
    my_bar.empty()

    # Show the results - with consistent border radius
    st.markdown(f"""
        <div style='background: #2a8c5e; padding: 1.5rem; border-radius: 15px;
        margin-top: 1.5rem; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
        <h2 style='color: white; margin-bottom: 1rem; text-align: center;'>🔍 Analysis Results</h2>
        <p style='font-size: 1.1rem; color: white; text-align: center;'>
        The plant <span style='font-weight: 600;'>
        {result['status']}</span> with <span style='font-weight: 600;'>{result['prediction']}</span> prediction.
        </p>
        </div>
        """, unsafe_allow_html=True)

