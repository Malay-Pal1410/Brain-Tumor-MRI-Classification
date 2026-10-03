import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuroVision AI | MRI Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS / CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_best.keras"
RESULTS_DIR = PROJECT_ROOT / "results"

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ============================================================
# PREMIUM UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- Base ---------- */
    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(65, 105, 225, 0.10), transparent 28rem),
            linear-gradient(180deg, #07101f 0%, #091426 45%, #07101f 100%);
        color: #edf3ff;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 16, 31, 0.72);
        backdrop-filter: blur(18px);
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2.1rem;
        padding-bottom: 4rem;
    }

    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, Helvetica, Arial, sans-serif;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081222 0%, #07101e 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.7rem;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08);
    }

    /* ---------- Brand ---------- */
    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.2rem;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        border-radius: 13px;
        display: grid;
        place-items: center;
        background: linear-gradient(145deg, #5d7cff, #8b5cf6);
        box-shadow: 0 10px 30px rgba(93,124,255,0.28);
        font-weight: 800;
        color: white;
    }

    .brand-title {
        color: #f8fbff;
        font-size: 1.05rem;
        font-weight: 750;
        line-height: 1.1;
    }

    .brand-sub {
        color: #8190aa;
        font-size: 0.72rem;
        margin-top: 4px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ---------- Hero ---------- */
    .hero {
        padding: 2.25rem 2.35rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 26px;
        background:
            linear-gradient(125deg, rgba(26,41,74,0.92), rgba(12,24,45,0.88));
        box-shadow: 0 24px 70px rgba(0,0,0,0.25);
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        right: -80px;
        top: -130px;
        border-radius: 50%;
        background: rgba(99,102,241,0.16);
        filter: blur(10px);
    }

    .eyebrow {
        color: #9fb1ff;
        font-size: 0.74rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .hero-title {
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.02;
        font-weight: 800;
        letter-spacing: -0.05em;
        max-width: 800px;
        margin-bottom: 0.85rem;
    }

    .hero-copy {
        color: #aebbd0;
        max-width: 780px;
        font-size: 1rem;
        line-height: 1.75;
    }

    /* ---------- Cards ---------- */
    .glass-card {
        background: rgba(13, 27, 50, 0.78);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 16px 45px rgba(0,0,0,0.16);
    }

    .mini-label {
        color: #7f91ad;
        font-size: 0.70rem;
        font-weight: 750;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .big-value {
        color: #f7faff;
        font-size: 1.65rem;
        font-weight: 780;
        letter-spacing: -0.035em;
    }

    .soft-copy {
        color: #91a0b8;
        font-size: 0.86rem;
        line-height: 1.6;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #46d79b;
        border-radius: 50%;
        margin-right: 7px;
        box-shadow: 0 0 0 5px rgba(70,215,155,0.10);
    }

    /* ---------- Result card ---------- */
    .result-card {
        border-radius: 22px;
        padding: 1.55rem;
        background:
            linear-gradient(145deg, rgba(18,36,66,0.98), rgba(11,25,47,0.98));
        border: 1px solid rgba(125,151,255,0.20);
        box-shadow: 0 18px 55px rgba(0,0,0,0.20);
        min-height: 160px;
    }

    .result-class {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.04em;
        margin: 0.2rem 0 0.4rem;
    }

    .confidence {
        color: #a9b8d0;
        font-size: 0.95rem;
    }

    /* ---------- Streamlit controls ---------- */
    div[data-testid="stFileUploader"] {
        background: rgba(13,27,50,0.58);
        border: 1px dashed rgba(143,162,205,0.32);
        border-radius: 18px;
        padding: 0.45rem;
    }

    div.stButton > button {
        border-radius: 12px;
        min-height: 46px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.08);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #5b73f2, #7758e8);
        color: white;
        box-shadow: 0 10px 28px rgba(91,115,242,0.25);
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
    }

    [data-testid="stMetric"] {
        background: rgba(13,27,50,0.72);
        border: 1px solid rgba(255,255,255,0.07);
        padding: 1rem 1.1rem;
        border-radius: 16px;
    }

    [data-testid="stExpander"] {
        background: rgba(13,27,50,0.52);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
    }

    /* ---------- Probability rows ---------- */
    .prob-row {
        display: flex;
        justify-content: space-between;
        color: #c8d3e5;
        font-size: 0.88rem;
        margin: 0.5rem 0 0.25rem;
    }

    /* ---------- Footer ---------- */
    .footer {
        color: #64748b;
        font-size: 0.76rem;
        text-align: center;
        padding: 2.5rem 0 0.5rem;
    }

    /* ---------- Hide default clutter ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_image(image):
    """
    The saved model already contains MobileNetV2 preprocess_input().
    Keep incoming pixels in the original 0-255 range.
    """
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.array(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def find_feature_extractor(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer
    return None


def generate_gradcam(model, image_array, class_index):
    """
    Working Grad-CAM implementation for the nested MobileNetV2 model.
    It intentionally avoids rebuilding a disconnected Functional graph.
    """
    feature_extractor = find_feature_extractor(model)
    if feature_extractor is None:
        raise ValueError("Could not locate MobileNetV2 feature extractor.")

    feature_index = None
    for index, layer in enumerate(model.layers):
        if layer.name == feature_extractor.name:
            feature_index = index
            break

    if feature_index is None:
        raise ValueError("MobileNetV2 layer position could not be determined.")

    x = tf.convert_to_tensor(image_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Run preprocessing / augmentation layers saved before MobileNetV2.
        for layer in model.layers[1:feature_index]:
            x = layer(x, training=False)

        feature_maps = feature_extractor(x, training=False)
        tape.watch(feature_maps)

        x = feature_maps
        for layer in model.layers[feature_index + 1:]:
            x = layer(x, training=False)

        predictions = x
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)
    if gradients is None:
        raise ValueError(
            "Grad-CAM gradients are None. Gradient path could not be computed."
        )

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    feature_maps = feature_maps[0]

    heatmap = tf.reduce_sum(
        feature_maps * pooled_gradients,
        axis=-1,
    )

    heatmap = tf.nn.relu(heatmap)
    maximum = tf.reduce_max(heatmap)

    if float(maximum) > 0:
        heatmap = heatmap / maximum

    return heatmap.numpy()


def resize_heatmap(heatmap):
    heatmap_tensor = tf.convert_to_tensor(
        heatmap[..., np.newaxis],
        dtype=tf.float32,
    )
    heatmap_tensor = tf.image.resize(
        heatmap_tensor,
        IMAGE_SIZE,
        method="bilinear",
    )
    resized = np.squeeze(heatmap_tensor.numpy())
    return np.clip(resized, 0.0, 1.0)


def create_gradcam_overlay(image, heatmap, alpha=0.40):
    original_image = image.convert("RGB").resize(IMAGE_SIZE)
    original_array = np.array(original_image, dtype=np.float32)

    colormap = plt.get_cmap("jet")
    colored_heatmap = colormap(heatmap)[..., :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.float32)

    overlay = (
        (1 - alpha) * original_array
        + alpha * colored_heatmap
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def section_header(eyebrow, title, copy):
    st.markdown(
        f"""
        <div style="margin:0.3rem 0 1.25rem;">
            <div class="eyebrow">{eyebrow}</div>
            <div style="font-size:2rem;font-weight:800;letter-spacing:-0.04em;
                        color:#f8fbff;margin-bottom:0.45rem;">{title}</div>
            <div class="soft-copy" style="max-width:800px;">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        """
        <div class="footer">
            NeuroVision AI · Research prototype · TensorFlow · MobileNetV2 · Grad-CAM
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL LOAD
# ============================================================

try:
    model = load_model()
    model_ready = True
except Exception as model_error:
    model_ready = False
    st.error("The trained MobileNetV2 model could not be loaded.")
    st.exception(model_error)
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand-wrap">
            <div class="brand-mark">N</div>
            <div>
                <div class="brand-title">NeuroVision AI</div>
                <div class="brand-sub">MRI Intelligence Platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("RESEARCH WORKSPACE")

    page = st.radio(
        "Navigation",
        ["Analyze", "Performance", "Dataset", "Methodology"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        """
        <div class="mini-label">Model</div>
        <div style="font-weight:700;color:#eef4ff;">MobileNetV2</div>
        <div class="soft-copy">Transfer-learning classifier</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    side_a, side_b = st.columns(2)
    with side_a:
        st.metric("Input", "224²")
    with side_b:
        st.metric("Classes", "4")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card" style="padding:0.9rem 1rem;">
            <div class="mini-label">System status</div>
            <div style="color:#dffbed;font-weight:700;">
                <span class="status-dot"></span>Model ready
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Academic research prototype · Not a clinical diagnostic system")


# ============================================================
# ANALYZE PAGE
# ============================================================

if page == "Analyze":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AI-ASSISTED MRI CLASSIFICATION</div>
            <div class="hero-title">Brain MRI analysis, built for clarity.</div>
            <div class="hero-copy">
                Classify a brain MRI across four supported categories and inspect
                the model's decision using Grad-CAM explainability. Designed as an
                engineering research interface—not a clinical diagnostic tool.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        st.metric("Architecture", "MobileNetV2")
    with stat2:
        st.metric("Input Resolution", "224 × 224")
    with stat3:
        st.metric("Output Classes", "4")
    with stat4:
        st.metric("Explainability", "Grad-CAM")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        section_header(
            "01 · INPUT",
            "Upload MRI",
            "Use a JPG, JPEG, or PNG brain MRI image. The image is resized to the model's 224 × 224 input resolution.",
        )

        uploaded_file = st.file_uploader(
            "Drop an MRI image here",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
        )

        image = None
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(
                    image,
                    caption="MRI preview",
                    use_container_width=True,
                )
            except Exception as image_error:
                st.error("The selected image could not be opened.")
                st.exception(image_error)

    with right:
        section_header(
            "02 · ANALYSIS",
            "Prediction workspace",
            "Run the trained classifier to generate class probabilities, confidence, and an explainability map.",
        )

        if image is None:
            st.markdown(
                """
                <div class="glass-card" style="min-height:250px;display:flex;
                     align-items:center;justify-content:center;text-align:center;">
                    <div>
                        <div style="font-size:2.2rem;margin-bottom:0.7rem;">◌</div>
                        <div style="font-weight:750;color:#eef4ff;font-size:1.05rem;">
                            Awaiting MRI input
                        </div>
                        <div class="soft-copy" style="margin-top:0.45rem;">
                            Upload an image to activate the analysis pipeline.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            analyze = st.button(
                "Run MRI Analysis",
                type="primary",
                use_container_width=True,
            )

            if analyze:
                try:
                    with st.spinner("Running MobileNetV2 inference..."):
                        processed_image = preprocess_image(image)
                        predictions = model.predict(
                            processed_image,
                            verbose=0,
                        )
                        probabilities = predictions[0]
                        predicted_index = int(np.argmax(probabilities))
                        predicted_class = CLASS_NAMES[predicted_index]
                        confidence = float(probabilities[predicted_index])

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="mini-label">Prediction</div>
                            <div class="result-class">{predicted_class}</div>
                            <div class="confidence">
                                Confidence · <strong style="color:#dfe7ff;">
                                {confidence * 100:.2f}%</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### Class distribution")

                    for class_name, probability in zip(
                        CLASS_NAMES,
                        probabilities,
                    ):
                        probability = float(probability)
                        st.markdown(
                            f"""
                            <div class="prob-row">
                                <span>{class_name}</span>
                                <span>{probability * 100:.2f}%</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.progress(
                            max(0.0, min(1.0, probability))
                        )

                    # Store for this rerun so Grad-CAM is shown immediately.
                    st.session_state["last_analysis"] = {
                        "predicted_index": predicted_index,
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "probabilities": probabilities.tolist(),
                        "predictions": predictions.tolist(),
                    }
                    st.session_state["analysis_image_bytes"] = uploaded_file.getvalue()

                except Exception as prediction_error:
                    st.error("The MRI analysis could not be completed.")
                    st.exception(prediction_error)

    # --------------------------------------------------------
    # GRAD-CAM / EXPLAINABILITY
    # --------------------------------------------------------

    if (
        image is not None
        and "last_analysis" in st.session_state
        and "analysis_image_bytes" in st.session_state
        and uploaded_file is not None
        and st.session_state["analysis_image_bytes"] == uploaded_file.getvalue()
    ):
        analysis = st.session_state["last_analysis"]
        predicted_index = analysis["predicted_index"]
        predicted_class = analysis["predicted_class"]
        confidence = analysis["confidence"]
        probabilities = np.array(analysis["probabilities"], dtype=np.float32)
        predictions = np.array(analysis["predictions"], dtype=np.float32)

        st.divider()

        section_header(
            "03 · EXPLAINABILITY",
            "Model attention with Grad-CAM",
            "Inspect the regions that contributed strongly to the classification decision. These activation maps explain model behavior; they are not tumor segmentation.",
        )

        try:
            with st.spinner("Generating Grad-CAM explanation..."):
                processed_image = preprocess_image(image)
                heatmap = generate_gradcam(
                    model,
                    processed_image,
                    predicted_index,
                )
                resized_heatmap = resize_heatmap(heatmap)
                overlay = create_gradcam_overlay(
                    image,
                    resized_heatmap,
                    alpha=0.40,
                )

            col1, col2, col3 = st.columns(3, gap="medium")

            with col1:
                st.markdown("##### Original MRI")
                st.image(
                    image,
                    caption="Source image",
                    use_container_width=True,
                )

            with col2:
                st.markdown("##### Activation map")
                fig, ax = plt.subplots()
                ax.imshow(
                    resized_heatmap,
                    cmap="jet",
                    vmin=0,
                    vmax=1,
                )
                ax.axis("off")
                fig.tight_layout(pad=0)
                st.pyplot(
                    fig,
                    use_container_width=True,
                )
                plt.close(fig)

            with col3:
                st.markdown("##### Explainability overlay")
                st.image(
                    overlay,
                    caption=f"{predicted_class} · {confidence * 100:.2f}%",
                    use_container_width=True,
                )

            st.info(
                "Red/yellow regions represent stronger positive contribution to "
                "the model's selected class. Grad-CAM is an explainability method "
                "and must not be interpreted as tumor segmentation or clinical localization."
            )

            with st.expander("Engineering details"):
                feature_extractor = find_feature_extractor(model)

                detail1, detail2, detail3 = st.columns(3)
                with detail1:
                    st.metric("Predicted index", predicted_index)
                with detail2:
                    st.metric("Confidence", f"{confidence * 100:.2f}%")
                with detail3:
                    st.metric(
                        "Feature extractor",
                        feature_extractor.name
                        if feature_extractor is not None
                        else "Unavailable",
                    )

                st.write("**Model input shape:**", model.input_shape)
                st.write("**Model output shape:**", model.output_shape)

                if feature_extractor is not None:
                    st.write(
                        "**Feature map shape:**",
                        feature_extractor.output.shape,
                    )

                st.write("**Raw prediction:**", predictions)

        except Exception as gradcam_error:
            st.warning(
                "Classification succeeded, but the Grad-CAM explanation "
                "could not be generated."
            )
            with st.expander("Grad-CAM error details"):
                st.exception(gradcam_error)

    st.warning(
        "Research use only — this application is an academic machine-learning "
        "prototype and is not intended to provide medical diagnosis or treatment advice."
    )

    footer()


# ============================================================
# PERFORMANCE PAGE
# ============================================================

elif page == "Performance":
    section_header(
        "MODEL INTELLIGENCE",
        "Performance",
        "A clean engineering view of the experiment artifacts already produced by the project.",
    )

    comparison = read_json(
        RESULTS_DIR / "model_comparison.json"
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Primary Model", "MobileNetV2")
    with m2:
        st.metric("Task", "4-Class")
    with m3:
        st.metric("Explainability", "Grad-CAM")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if comparison:
        st.markdown("### Model comparison artifact")
        st.json(comparison)
    else:
        st.info(
            "The interface could not read results/model_comparison.json. "
            "When that experiment artifact is available, its results can be "
            "presented here without changing the prediction pipeline."
        )

    history_files = [
        ("Custom CNN", RESULTS_DIR / "custom_cnn_test_metrics.json"),
        ("EfficientNet", RESULTS_DIR / "efficientnet_test_metrics.json"),
        ("MobileNetV2", RESULTS_DIR / "mobilenet_test_metrics.json"),
    ]

    st.markdown("### Saved evaluation artifacts")
    cols = st.columns(3)

    for col, (name, path) in zip(cols, history_files):
        with col:
            data = read_json(path)
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="mini-label">Experiment</div>
                    <div class="big-value" style="font-size:1.2rem;">{name}</div>
                    <div class="soft-copy">
                        {"Metrics artifact detected." if data else "Metrics artifact not available."}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if data:
                with st.expander(f"View {name} metrics"):
                    st.json(data)

    footer()


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "Dataset":
    section_header(
        "DATA PIPELINE",
        "Dataset",
        "The project uses separate training, validation, and test directories for four-class brain MRI classification.",
    )

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Training Images", "1,695")
    with d2:
        st.metric("Validation Images", "502")
    with d3:
        st.metric("Test Images", "246")
    with d4:
        st.metric("Classes", "4")

    st.markdown("### Supported categories")

    class_cols = st.columns(4)
    descriptions = [
        ("Glioma", "Brain MRI classification category"),
        ("Meningioma", "Brain MRI classification category"),
        ("No Tumor", "Non-tumor classification category"),
        ("Pituitary", "Brain MRI classification category"),
    ]

    for col, (name, description) in zip(class_cols, descriptions):
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="mini-label">Class</div>
                    <div class="big-value" style="font-size:1.18rem;">{name}</div>
                    <div class="soft-copy">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Input pipeline")
    st.markdown(
        """
        <div class="glass-card">
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:18px;">
                <div><div class="mini-label">Resolution</div><b>224 × 224 RGB</b></div>
                <div><div class="mini-label">Batch size</div><b>32</b></div>
                <div><div class="mini-label">Label mode</div><b>Categorical</b></div>
                <div><div class="mini-label">Seed</div><b>42</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Training uses augmentation; validation and test data are evaluated "
        "without training-time augmentation."
    )

    footer()


# ============================================================
# METHODOLOGY PAGE
# ============================================================

elif page == "Methodology":
    section_header(
        "ENGINEERING NOTES",
        "Methodology",
        "How the current application moves from an uploaded MRI to a classification and an explainability visualization.",
    )

    st.markdown("### Inference pipeline")

    steps = [
        ("01", "Image input", "MRI is converted to RGB and resized to 224 × 224."),
        (
            "02",
            "Model preprocessing",
            "Pixel values remain in the 0–255 range at app level because MobileNetV2 preprocessing is already embedded inside the saved model.",
        ),
        (
            "03",
            "Feature extraction",
            "The trained MobileNetV2 backbone produces deep image features.",
        ),
        (
            "04",
            "Classification",
            "The network outputs probabilities for Glioma, Meningioma, No Tumor, and Pituitary.",
        ),
        (
            "05",
            "Grad-CAM",
            "Gradients are combined with MobileNetV2 feature maps to visualize regions contributing to the selected class.",
        ),
    ]

    for number, title, copy in steps:
        st.markdown(
            f"""
            <div class="glass-card" style="margin-bottom:12px;">
                <div style="display:flex;gap:18px;align-items:flex-start;">
                    <div style="min-width:42px;height:42px;border-radius:12px;
                         display:grid;place-items:center;background:rgba(99,102,241,.16);
                         color:#aebaff;font-weight:800;">{number}</div>
                    <div>
                        <div style="font-weight:760;color:#f2f6ff;font-size:1rem;">
                            {title}
                        </div>
                        <div class="soft-copy" style="margin-top:5px;">{copy}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Technology stack")
    t1, t2, t3, t4 = st.columns(4)
    for col, title, value in [
        (t1, "Framework", "TensorFlow / Keras"),
        (t2, "Interface", "Streamlit"),
        (t3, "Backbone", "MobileNetV2"),
        (t4, "XAI", "Grad-CAM"),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="mini-label">{title}</div>
                    <div class="big-value" style="font-size:1.05rem;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.warning(
        "Grad-CAM visualizes model attention/contribution. It does not provide "
        "tumor segmentation, clinical localization, or a medical diagnosis."
    )

    footer()
