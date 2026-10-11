import streamlit as st
import tensorflow as tf
import numpy as np
import base64

from PIL import Image
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BrainTumor AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "brain_tumor_mobilenetv2_final.h5"

BRAIN_IMAGE_PATH = BASE_DIR / "assets" / "brain_ai_visual.png"


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    html,
    body,
    [class*="css"] {
        font-family: "Segoe UI", Arial, sans-serif;
    }

    .stApp {

        background:
            radial-gradient(
                circle at 70% 12%,
                rgba(20, 115, 145, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 10% 80%,
                rgba(12, 70, 90, 0.07),
                transparent 30%
            ),
            #05090f;

        color: #edf4f8;
    }

    .block-container {

        max-width: 1050px;

        padding-top: 45px;
        padding-bottom: 60px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .header-content {
        padding-top: 35px;
        padding-left: 5px;
    }

    .brand-name {

        font-size: 15px;

        font-weight: 600;

        letter-spacing: 2px;

        color: #27d9ff;

        margin-bottom: 15px;
    }

    .page-title {

        font-size: 44px;

        font-weight: 700;

        line-height: 1.1;

        color: #f4f8fa;

        letter-spacing: -1px;
    }

    .page-subtitle {

        margin-top: 14px;

        font-size: 16px;

        color: #8b9aa6;

        line-height: 1.6;

        max-width: 550px;
    }

    .system-status {

        margin-top: 21px;

        display: flex;

        align-items: center;

        gap: 8px;

        font-size: 11px;

        font-weight: 600;

        letter-spacing: 1.5px;

        color: #91a5b1;
    }

    .status-dot {

        width: 7px;

        height: 7px;

        display: inline-block;

        border-radius: 50%;

        background: #27d9ff;

        box-shadow:
            0 0 6px rgba(39, 217, 255, 0.9),
            0 0 15px rgba(39, 217, 255, 0.45);

        animation: pulse 2.5s ease-in-out infinite;
    }

    @keyframes pulse {

        0%,
        100% {
            opacity: 0.55;
            transform: scale(0.9);
        }

        50% {
            opacity: 1;
            transform: scale(1);
        }
    }


    /* ========================================================
       BRAIN IMAGE AREA
       ======================================================== */

   /* ========================================================
   HERO SECTION
   ======================================================== */

.hero-section {

    position: relative;

    min-height: 330px;

    width: 100%;

    display: flex;

    align-items: flex-start;

    overflow: hidden;
}


.hero-content {

    position: relative;

    z-index: 2;

    width: 58%;

    padding-top: 35px;

    padding-left: 5px;
}


.hero-visual {

    position: absolute;

    right: 20px;

    top: 5px;

    width: 42%;

    height: 310px;

    display: flex;

    align-items: center;

    justify-content: center;

    z-index: 1;
}


.hero-visual::before {

    content: "";

    position: absolute;

    width: 260px;

    height: 260px;

    border-radius: 50%;

    background: rgba(0, 195, 255, 0.08);

    filter: blur(65px);

}


.hero-visual img {

    position: relative;

    z-index: 2;

    width: 330px;

    max-width: 100%;

    height: auto;

    object-fit: contain;

    filter:
        drop-shadow(0 0 12px rgba(0, 205, 255, 0.22))
        drop-shadow(0 0 35px rgba(0, 180, 255, 0.10));
}

   


    /* ========================================================
       UPLOAD CARD
       ======================================================== */

    .upload-card {

        padding: 28px;

        border-radius: 18px;

        border:
            1px solid
            rgba(105, 165, 185, 0.15);

        background:
            linear-gradient(
                145deg,
                rgba(16, 29, 38, 0.90),
                rgba(7, 13, 20, 0.95)
            );

        box-shadow:
            0 20px 55px rgba(0, 0, 0, 0.28);

        margin-top: 12px;

        margin-bottom: 14px;
    }

    .section-label {

        font-size: 11px;

        font-weight: 650;

        letter-spacing: 2px;

        color: #62c7df;

        margin-bottom: 10px;
    }

    .card-title {

        font-size: 20px;

        font-weight: 600;

        color: #e9f1f4;

        margin-bottom: 7px;
    }

    .muted {

        color: #81929d;

        font-size: 14px;

        line-height: 1.7;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {

        margin-top: 0;
    }

    [data-testid="stFileUploaderDropzone"] {

        background:
            rgba(7, 15, 22, 0.75) !important;

        border:
            1px dashed
            rgba(73, 170, 198, 0.32) !important;

        border-radius:
            14px !important;

        min-height:
            125px !important;

        transition:
            all 0.25s ease !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {

        background:
            rgba(10, 23, 31, 0.85) !important;

        border-color:
            rgba(80, 202, 230, 0.55) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {

        color: #83939e !important;
    }

    [data-testid="stFileUploader"] button {

        border-radius:
            9px !important;

        border:
            1px solid
            rgba(81, 187, 215, 0.35) !important;

        background:
            rgba(16, 42, 54, 0.75) !important;

        color:
            #dceff4 !important;
    }


    /* ========================================================
       RESULT CARD
       ======================================================== */

    .result-card {

        margin-top: 30px;

        padding: 28px;

        min-height: 180px;

        border-radius: 18px;

        border:
            1px solid
            rgba(91, 186, 211, 0.17);

        background:
            linear-gradient(
                145deg,
                rgba(15, 30, 39, 0.90),
                rgba(7, 13, 20, 0.96)
            );

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.30);
    }

    .prediction-label {

        color: #718a95;

        font-size: 11px;

        letter-spacing: 2px;

        font-weight: 650;

        margin-bottom: 9px;
    }

    .prediction-name {

        color: #f1f7f9;

        font-size: 34px;

        font-weight: 700;

        letter-spacing: -0.8px;

        margin-bottom: 7px;
    }

    .confidence {

        color: #70d1e4;

        font-size: 15px;

        font-weight: 550;

        margin-bottom: 12px;
    }

    .result-description {

        color: #84959f;

        font-size: 14px;

        line-height: 1.65;
    }


    /* ========================================================
       PREVIEW
       ======================================================== */

    .preview-card {

        margin-top: 30px;

        padding: 20px;

        border-radius: 18px;

        border:
            1px solid
            rgba(91, 186, 211, 0.17);

        background:
            rgba(9, 17, 24, 0.80);
    }

    .preview-title {

        color: #718791;

        font-size: 11px;

        letter-spacing: 1.8px;

        font-weight: 650;

        margin-bottom: 12px;
    }


    /* ========================================================
       BREAKDOWN
       ======================================================== */

    .breakdown-card {

        margin-top: 20px;

        padding: 25px 28px;

        border-radius: 17px;

        border:
            1px solid
            rgba(115, 170, 190, 0.12);

        background:
            rgba(9, 17, 24, 0.78);
    }

    .breakdown-title {

        color: #dce7eb;

        font-size: 16px;

        font-weight: 600;

        margin-bottom: 22px;
    }

    .probability-row {

        margin-bottom: 18px;
    }

    .probability-header {

        display: flex;

        justify-content: space-between;

        margin-bottom: 7px;
    }

    .probability-name {

        color: #aabac2;

        font-size: 13px;
    }

    .probability-value {

        color: #d8e6ea;

        font-size: 12px;

        font-weight: 600;
    }

    .bar-background {

        height: 5px;

        width: 100%;

        border-radius: 10px;

        background:
            rgba(110, 150, 165, 0.12);

        overflow: hidden;
    }

    .bar-fill {

        height: 100%;

        border-radius: 10px;

        background:
            linear-gradient(
                90deg,
                #327d93,
                #67c9df
            );
    }


    /* ========================================================
       MODEL PERFORMANCE
       ======================================================== */

    .performance-title {

        margin-top: 42px;

        margin-bottom: 18px;

        font-size: 11px;

        letter-spacing: 2px;

        font-weight: 650;

        color: #708791;
    }

    .metric-card {

        padding: 21px;

        min-height: 105px;

        border-radius: 15px;

        border:
            1px solid
            rgba(117, 165, 181, 0.12);

        background:
            rgba(10, 18, 25, 0.72);

        transition:
            transform 0.25s ease,
            border-color 0.25s ease;
    }

    .metric-card:hover {

        transform:
            translateY(-2px);

        border-color:
            rgba(85, 186, 211, 0.25);
    }

    .metric-name {

        color: #8b9aa2;

        font-size: 12px;

        margin-bottom: 9px;
    }

    .metric-value {

        color: #eaf4f7;

        font-size: 24px;

        font-weight: 650;
    }

    .metric-selected {

        margin-top: 7px;

        color: #69cadf;

        font-size: 10px;

        letter-spacing: 1.2px;

        font-weight: 600;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton {

        margin-top: 30px;
    }

    .stButton > button {

        width: 100%;

        min-height: 45px;

        border-radius: 10px;

        border:
            1px solid
            rgba(93, 190, 215, 0.32);

        background:
            rgba(18, 42, 52, 0.72);

        color: #dceff4;

        font-size: 13px;

        font-weight: 600;

        transition: all 0.25s ease;
    }

    .stButton > button:hover {

        border-color:
            rgba(100, 210, 232, 0.65);

        background:
            rgba(25, 60, 73, 0.88);

        color: #f1fbfd;

        transform:
            translateY(-1px);
    }


    /* ========================================================
       DISCLAIMER
       ======================================================== */

    .disclaimer {

        margin-top: 42px;

        padding: 16px 20px;

        border-left:
            2px solid
            rgba(94, 180, 201, 0.35);

        background:
            rgba(10, 18, 25, 0.55);

        color: #697b85;

        font-size: 11px;

        line-height: 1.6;

        border-radius:
            0 8px 8px 0;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-top: 25px;
        }

        .page-title {
            font-size: 34px;
        }

        .brain-wrapper {
            height: 230px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    loaded_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return loaded_model


try:

    model = load_model()

except Exception as error:

    st.error(
        "Unable to load the trained model."
    )

    st.code(
        str(error)
    )

    st.stop()

# ============================================================
# HERO SECTION
# ============================================================

if BRAIN_IMAGE_PATH.exists():

    with open(BRAIN_IMAGE_PATH, "rb") as image_file:

        brain_base64 = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

else:

    brain_base64 = ""


st.html(
    f"""
    <div class="hero-section">

        <div class="hero-content">

            <div class="brand-name">
                BrainTumor AI
            </div>

            <div class="page-title">
                Brain MRI Analysis
            </div>

            <div class="page-subtitle">
                Upload an MRI scan and let the AI analyze the image.
            </div>

            <div class="system-status">

                <span class="status-dot"></span>

                AI SYSTEM ONLINE

            </div>

        </div>


        <div class="hero-visual">

            {
                f'<img src="data:image/png;base64,{brain_base64}" />'
                if brain_base64
                else ""
            }

        </div>

    </div>
    """
)


# ============================================================
# UPLOAD INFORMATION CARD
# ============================================================

st.html(
    """
    <div class="upload-card">

        <div class="section-label">
            UPLOAD MRI SCAN
        </div>

        <div class="card-title">
            Select an MRI image
        </div>

        <div class="muted">

            Drag &amp; drop your MRI image here
            or browse your files.

            <br>

            Supported formats:
            JPG, JPEG, PNG.

        </div>

    </div>
    """
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "MRI image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ----------------------------------------------------
        # PREPARE IMAGE
        # ----------------------------------------------------

        image_resized = image.resize(
            (224, 224)
        )

        image_array = np.asarray(
            image_resized,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        #
        # The final .h5 model already contains:
        #
        # Rescaling(1./127.5, offset=-1)
        #
        # Therefore raw 0-255 pixel values are passed here.
        # ----------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = (
            float(predictions[predicted_index])
            * 100
        )


        # ====================================================
        # RESULT SECTION
        # ====================================================

        result_col, preview_col = st.columns(
            [1.05, 0.95],
            gap="large"
        )


        # ====================================================
        # RESULT
        # ====================================================

        with result_col:

            st.html(
                f"""
                <div class="result-card">

                    <div class="prediction-label">
                        ANALYSIS RESULT
                    </div>

                    <div class="prediction-name">
                        {predicted_class}
                    </div>

                    <div class="confidence">
                        {confidence:.1f}% Model Confidence
                    </div>

                    <div class="result-description">

                        The model's highest-confidence
                        classification is
                        <strong>{predicted_class}</strong>.

                    </div>

                </div>
                """
            )


        # ====================================================
        # MRI PREVIEW
        # ====================================================

        with preview_col:

            st.html(
                """
                <div class="preview-card">

                    <div class="preview-title">
                        MRI PREVIEW
                    </div>

                </div>
                """
            )

            st.image(
                image,
                use_container_width=True
            )


        # ====================================================
        # PREDICTION BREAKDOWN
        # ====================================================

        st.html(
            """
            <div class="breakdown-card">

                <div class="breakdown-title">
                    Prediction Breakdown
                </div>
            """
        )


        for class_name, probability in zip(
            CLASS_NAMES,
            predictions
        ):

            percentage = (
                float(probability)
                * 100
            )

            st.html(
                f"""
                <div class="probability-row">

                    <div class="probability-header">

                        <span class="probability-name">
                            {class_name}
                        </span>

                        <span class="probability-value">
                            {percentage:.1f}%
                        </span>

                    </div>

                    <div class="bar-background">

                        <div
                            class="bar-fill"
                            style="width: {percentage:.2f}%;">
                        </div>

                    </div>

                </div>
                """
            )


        st.html(
            """
            </div>
            """
        )


        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        st.html(
            """
            <div class="performance-title">
                MODEL PERFORMANCE
            </div>
            """
        )


        model_col1, model_col2, model_col3 = st.columns(
            3,
            gap="medium"
        )


        # ----------------------------------------------------
        # CUSTOM CNN
        # ----------------------------------------------------

        with model_col1:

            st.html(
                """
                <div class="metric-card">

                    <div class="metric-name">
                        Custom CNN
                    </div>

                    <div class="metric-value">
                        80.48%
                    </div>

                </div>
                """
            )


        # ----------------------------------------------------
        # MOBILENETV2
        # ----------------------------------------------------

        with model_col2:

            st.html(
                """
                <div class="metric-card">

                    <div class="metric-name">
                        MobileNetV2
                    </div>

                    <div class="metric-value">
                        82.52%
                    </div>

                </div>
                """
            )


        # ----------------------------------------------------
        # FINE-TUNED MOBILENETV2
        # ----------------------------------------------------

        with model_col3:

            st.html(
                """
                <div class="metric-card">

                    <div class="metric-name">
                        Fine-Tuned MobileNetV2
                    </div>

                    <div class="metric-value">
                        88.21%
                    </div>

                    <div class="metric-selected">
                        SELECTED MODEL
                    </div>

                </div>
                """
            )


        # ====================================================
        # ANALYZE ANOTHER SCAN
        # ====================================================

        if st.button(
            "ANALYZE ANOTHER SCAN",
            use_container_width=True
        ):

            st.rerun()


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        st.error(
            "Unable to analyze the uploaded image."
        )

        st.code(
            str(error)
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.html(
    """
    <div class="disclaimer">

        This application is an educational AI prototype
        and is not intended to replace professional
        medical diagnosis.

    </div>
    """
)