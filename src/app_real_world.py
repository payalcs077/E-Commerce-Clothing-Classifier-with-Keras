from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from real_world_inference import (
    dataset_status,
    load_real_world_labels,
    load_real_world_model,
    predict_pil_image,
)

st.set_page_config(
    page_title="Clothing Classifier",
    page_icon="👕",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4efe6 0%, #fffaf4 100%);
        color: #1f2933;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    .app-shell {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
    }
    .metric-card {
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 16px;
        padding: 1rem;
        background: #fffdf9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_model():
    return load_real_world_model()


@st.cache_data
def cached_labels():
    return load_real_world_labels()


def model_is_ready() -> bool:
    try:
        cached_model()
        cached_labels()
        return True
    except FileNotFoundError:
        return False


def render_sidebar() -> None:
    counts = dataset_status()
    st.sidebar.header("Dataset Status")
    st.sidebar.write(f"Raw images: `{counts['raw']}`")
    st.sidebar.write(f"Train images: `{counts['train']}`")
    st.sidebar.write(f"Validation images: `{counts['val']}`")
    st.sidebar.write(f"Test images: `{counts['test']}`")

    st.sidebar.header("Run Order")
    st.sidebar.code(
        "\n".join(
            [
                "python src/prepare_real_data.py",
                "python src/train_real_world.py",
                "streamlit run src/app_real_world.py",
            ]
        )
    )


def render_prediction(uploaded_file) -> None:
    image = Image.open(BytesIO(uploaded_file.getvalue()))
    top_label, top_score, ranked = predict_pil_image(image)

    left_column, right_column = st.columns([1.1, 0.9], gap="large")

    with left_column:
        st.image(image, caption="Uploaded product image", use_container_width=True)

    with right_column:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Prediction")
        st.metric("Top class", top_label)
        st.metric("Confidence", f"{top_score:.2%}")
        st.markdown("</div>", unsafe_allow_html=True)

        scores = pd.DataFrame(ranked, columns=["Class", "Probability"])
        scores["Probability"] = scores["Probability"].round(4)
        st.subheader("Class Probabilities")
        st.dataframe(scores, use_container_width=True, hide_index=True)


def main() -> None:
    render_sidebar()

    st.title("E-Commerce Clothing Classifier")
    st.write(
        "Upload a product image to classify it with the directory-based transfer-learning model."
    )

    if not model_is_ready():
        st.warning(
            "Real-world model not found. Add images to data/raw, run "
            "`python src/prepare_real_data.py`, then `python src/train_real_world.py`."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a clothing image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is None:
        st.info("Upload one image to run inference.")
        st.stop()

    st.markdown('<div class="app-shell">', unsafe_allow_html=True)
    render_prediction(uploaded_file)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

