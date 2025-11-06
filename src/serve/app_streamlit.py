"""Streamlit web UI for plant disease detection."""

import os
import sys
from pathlib import Path
from io import BytesIO

import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.inference import PlantDiseasePredictor
from src.explainability.gradcam import GradCAM
from src.data.augmentations import load_and_preprocess_image
try:
    # Optional helper to download model if hosted remotely
    from src.utils.model_downloader import (
        ensure_model_available,
        get_model_path_from_secrets,
        get_download_config,
    )
except Exception:
    ensure_model_available = None
    get_model_path_from_secrets = None
    get_download_config = None


# Page config
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .healthy {
        border-left: 5px solid #4CAF50;
    }
    .diseased {
        border-left: 5px solid #F44336;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor(model_path, class_names_path):
    """Load predictor model (cached)."""
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    predictor = PlantDiseasePredictor(
        model_path=model_path,
        class_names=class_names,
        device=device
    )
    
    return predictor


def resolve_model_and_classnames():
    """Resolve model and class names paths from secrets/env/fallbacks."""
    # 1) Secrets (Streamlit Cloud/local secrets)
    secret_model_path = None
    try:
        if hasattr(st, "secrets") and "model" in st.secrets:
            secret_model_path = st.secrets["model"].get("MODEL_PATH")
    except Exception:
        pass

    # 2) Environment variables
    env_model_path = os.environ.get("MODEL_PATH", "").strip()
    env_model_dir = os.environ.get("MODEL_DIR", "").strip()

    # 3) Fallback candidates (common locations)
    fallback_models = [
        "models/streamlit_run/best_model.pth",
        "models/demo_run/best_model.pth",
        "models/best_model.pth",
    ]

    candidates = [env_model_path, secret_model_path] + fallback_models
    candidates = [c for c in candidates if c]

    chosen_model = None
    for c in candidates:
        p = Path(c)
        # Make relative to project root
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists():
            chosen_model = str(p)
            break

    # If still not found, optionally try downloading via helper
    if chosen_model is None and ensure_model_available and get_model_path_from_secrets:
        try:
            target_path = get_model_path_from_secrets()
            cfg = get_download_config()
            resolved = ensure_model_available(target_path or fallback_models[0], **cfg)
            if resolved and resolved.exists():
                chosen_model = str(resolved)
        except Exception:
            pass

    if not chosen_model:
        return None, None

    model_dir = Path(chosen_model).parent
    # Prefer explicit env MODEL_DIR if present and valid
    if env_model_dir:
        env_dir = Path(env_model_dir)
        if env_dir.exists():
            model_dir = env_dir

    class_names_path = model_dir / "class_names.txt"
    return str(chosen_model), str(class_names_path)


def process_image(image, predictor, generate_gradcam=True):
    """Process uploaded image."""
    # Save temp file
    temp_path = 'temp_image.jpg'
    image.save(temp_path)
    
    # Predict
    result = predictor.predict(temp_path, topk=3)
    
    # Generate Grad-CAM if requested
    gradcam_overlay = None
    if generate_gradcam:
        try:
            preprocessed, original = load_and_preprocess_image(temp_path)
            input_tensor = preprocessed.unsqueeze(0).to(predictor.device)
            
            gradcam = GradCAM(predictor.model)
            gradcam_overlay = gradcam.overlay_heatmap(original, input_tensor, alpha=0.4)
        except Exception as e:
            st.warning(f"Could not generate Grad-CAM: {e}")
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return result, gradcam_overlay


def main():
    # Header
    st.markdown('<h1 class="main-header">🌱 Plant Disease Detector</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This tool uses deep learning to detect plant diseases from leaf images.
        
        **Features:**
        - Multi-class disease detection
        - Confidence scores
        - Top-3 predictions
        - Grad-CAM explainability
        """)
        
        st.header("⚙️ Settings")
        show_gradcam = st.checkbox("Show Grad-CAM", value=True)
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.0, 0.05)
        
        st.header("📊 Model Info")
        
        # Resolve model and class names
        model_path, class_names_path = resolve_model_and_classnames()

        if model_path and os.path.exists(model_path):
            st.success("✓ Model loaded")
            st.text(f"Path: {os.path.relpath(model_path, Path.cwd())}")

            # Device info
            device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
            st.text(f"Device: {device}")
        else:
            st.error("⚠️ Model not found. Set MODEL_PATH in .streamlit/secrets.toml or place a model at one of: \n- models/streamlit_run/best_model.pth\n- models/demo_run/best_model.pth")
            st.stop()
    
    # Load predictor
    # class_names_path resolved alongside model
    if not class_names_path or not os.path.exists(class_names_path):
        st.error("⚠️ Class names file not found!")
        st.stop()
    
    try:
        predictor = load_predictor(model_path, class_names_path)
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📷 Single Image", "📁 Batch Processing", "📖 Help"])
    
    with tab1:
        st.header("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a plant leaf"
        )
        
        # Sample images (if available)
        st.subheader("Or try a sample image:")
        col1, col2, col3 = st.columns(3)
        sample_clicked = None
        
        # Placeholder for sample images
        with col1:
            if st.button("🌿 Sample 1"):
                sample_clicked = 1
        with col2:
            if st.button("🍃 Sample 2"):
                sample_clicked = 2
        with col3:
            if st.button("🌾 Sample 3"):
                sample_clicked = 3
        
        # Process image
        if uploaded_file is not None or sample_clicked:
            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
            else:
                st.info("Sample images not implemented in this demo")
                st.stop()
            
            # Display image
            st.subheader("Input Image")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict
            with st.spinner("Analyzing image..."):
                result, gradcam_overlay = process_image(image, predictor, show_gradcam)
            
            # Display results
            st.subheader("🎯 Prediction Results")
            
            # Top prediction
            top_pred = result['predictions'][0]
            is_healthy = 'healthy' in top_pred['label'].lower()
            
            box_class = 'healthy' if is_healthy else 'diseased'
            
            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h2>{top_pred['label'].replace('_', ' ').title()}</h2>
                <h3>Confidence: {top_pred['probability']:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Uncertainty warning
            if result['uncertain']:
                st.warning("⚠️ Low confidence prediction. Consider retaking the image or consulting an expert.")
            
            # Top-3 predictions
            st.subheader("Top 3 Predictions")
            
            for i, pred in enumerate(result['predictions'], 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{i}. **{pred['label'].replace('_', ' ').title()}**")
                with col2:
                    st.metric("", f"{pred['probability']:.1%}")
                
                st.progress(pred['probability'])
            
            # Grad-CAM visualization
            if show_gradcam and gradcam_overlay is not None:
                with col2:
                    st.subheader("Explainability (Grad-CAM)")
                    st.image(gradcam_overlay, caption="Attention Heatmap", use_column_width=True)
                    st.caption("Highlighted regions show areas the model focused on for its prediction.")
            
            # Download results
            st.subheader("📥 Download Results")
            
            # JSON download
            import json
            result_json = json.dumps(result, indent=2)
            st.download_button(
                label="Download JSON",
                data=result_json,
                file_name="prediction_result.json",
                mime="application/json"
            )
    
    with tab2:
        st.header("Batch Processing")
        st.info("Batch processing is available via CLI. Use: `python src/cli.py batch_predict --help`")
        
        st.code("""
# Example batch processing command:
python src/cli.py batch_predict \\
    --model ./models/best_model.pth \\
    --input-dir ./images/ \\
    --output-dir ./predictions/ \\
    --explain
        """, language="bash")
    
    with tab3:
        st.header("📖 How to Use")
        
        st.markdown("""
        ### Steps:
        1. **Upload an image** of a plant leaf using the file uploader
        2. **Wait for analysis** - the model will process the image
        3. **Review predictions** - see the top-3 most likely disease classes
        4. **Check Grad-CAM** - visualize which parts of the leaf influenced the prediction
        5. **Download results** - export predictions as JSON for record-keeping
        
        ### Tips for Best Results:
        - 📸 Use clear, well-lit images
        - 🔍 Focus on a single leaf when possible
        - 🌿 Ensure the leaf fills most of the frame
        - 📏 Recommended image size: at least 224x224 pixels
        - 🎨 Avoid extreme lighting or heavy shadows
        
        ### Disease Classes:
        """)
        
        # Display class names
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
        
        for i, class_name in enumerate(class_names, 1):
            st.write(f"{i}. {class_name.replace('_', ' ').title()}")
        
        st.markdown("""
        ### About Grad-CAM:
        Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions 
        of the image that were most important for the model's prediction. Red/warm 
        colors indicate high importance, while blue/cool colors indicate low importance.
        
        ### Support:
        For issues or questions, please refer to the project documentation or contact support.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Plant Disease Detector v1.0 | Built with PyTorch & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
