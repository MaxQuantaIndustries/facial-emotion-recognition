import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
import cv2
from keras.models import load_model
from PIL import Image

# Load the model and define labels
import os
import gdown

MODEL_PATH = "best_model.h5"
GOOGLE_DRIVE_FILE_ID = "1d0bBWGSXWiQqFDGyi29u8dI0KV73sxJM"  # Replace with your actual file ID

if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

from keras.models import load_model
model = load_model(MODEL_PATH, compile=False)

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise', 'disgust', 'fear']
img_size = (224, 224)

# Title
st.title("Facial Emotion Recognition App")

# Tabs
tab1, tab2 = st.tabs(["🙂 Facial", "📊 Analytics"])

# ---------------------------------------------
# Tab 1: Facial Emotion Prediction
# ---------------------------------------------
with tab1:
    st.header("Facial Emotion Detection")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess image
        img_array = np.array(image)
        resized_img = cv2.resize(img_array, img_size)
        normalized_img = resized_img / 255.0
        input_img = np.expand_dims(normalized_img, axis=0)

        # Predict
        prediction = model.predict(input_img)
        predicted_class = class_labels[np.argmax(prediction)]

        st.subheader("Prediction:")
        st.success(f"Emotion: **{predicted_class}**")
        st.bar_chart(prediction[0])

# ---------------------------------------------
# Tab 2: Model Analytics
# ---------------------------------------------
with tab2:
    st.header("Model Performance Analytics")

    # Classification report data
    report_data = {
        "Class": ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise', 'disgust', 'fear'],
        "Precision": [0.91, 0.83, 0.78, 0.90, 0.84, 1.00, 0.97],
        "Recall": [0.82, 0.99, 0.83, 0.74, 0.93, 0.95, 0.86],
        "F1-score": [0.87, 0.90, 0.80, 0.81, 0.88, 0.98, 0.92],
        "Support": [312, 314, 317, 327, 328, 109, 162]
    }
    df = pd.DataFrame(report_data)

    st.subheader("📄 Classification Report")
    st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'))

    # Confusion matrix
    st.subheader("🔷 Confusion Matrix")
    cm = np.array([
        [257, 8, 13, 12, 19, 0, 3],
        [0, 310, 3, 1, 0, 0, 0],
        [5, 23, 263, 13, 13, 0, 0],
        [13, 11, 52, 241, 9, 0, 1],
        [3, 13, 7, 1, 304, 0, 0],
        [2, 3, 0, 0, 0, 104, 0],
        [2, 4, 0, 0, 16, 0, 140]
    ])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Plotly bar chart for metrics
    st.subheader("📈 Metrics Comparison")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=report_data["Class"], y=report_data["Precision"], name='Precision'))
    fig.add_trace(go.Bar(x=report_data["Class"], y=report_data["Recall"], name='Recall'))
    fig.add_trace(go.Bar(x=report_data["Class"], y=report_data["F1-score"], name='F1-score'))

    fig.update_layout(
        barmode='group',
        xaxis_title='Classes',
        yaxis_title='Score',
        title='Precision, Recall, F1-score Comparison'
    )
    st.plotly_chart(fig)
