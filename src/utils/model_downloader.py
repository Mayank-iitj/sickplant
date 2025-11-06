"""
Model Download Utility for Streamlit Cloud
Downloads model from external URL if not present locally
"""
import os
from pathlib import Path
from typing import Optional
import streamlit as st


def download_from_url(url: str, save_path: Path) -> bool:
    """
    Download file from URL with progress bar
    
    Args:
        url: URL to download from
        save_path: Local path to save file
        
    Returns:
        bool: True if successful
    """
    try:
        import requests
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with st.spinner(f"Downloading model from {url}..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            progress_bar = st.progress(0)
            downloaded = 0
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
            
            progress_bar.progress(1.0)
            st.success(f"✅ Model downloaded successfully to {save_path}")
            return True
            
    except Exception as e:
        st.error(f"❌ Failed to download model: {str(e)}")
        return False


def download_from_google_drive(file_id: str, save_path: Path) -> bool:
    """
    Download file from Google Drive
    
    Args:
        file_id: Google Drive file ID
        save_path: Local path to save file
        
    Returns:
        bool: True if successful
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return download_from_url(url, save_path)


def download_from_huggingface(repo_id: str, filename: str, save_path: Path) -> bool:
    """
    Download file from Hugging Face Hub
    
    Args:
        repo_id: Hugging Face repo ID (e.g., "username/model-name")
        filename: File name in the repo
        save_path: Local path to save file
        
    Returns:
        bool: True if successful
    """
    try:
        from huggingface_hub import hf_hub_download
        
        with st.spinner(f"Downloading from Hugging Face: {repo_id}/{filename}..."):
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=save_path.parent
            )
            
            # Move to desired location
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if Path(downloaded_path) != save_path:
                import shutil
                shutil.copy(downloaded_path, save_path)
            
            st.success(f"✅ Model downloaded from Hugging Face")
            return True
            
    except Exception as e:
        st.error(f"❌ Failed to download from Hugging Face: {str(e)}")
        return False


def ensure_model_available(
    model_path: str,
    model_url: Optional[str] = None,
    gdrive_id: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_filename: Optional[str] = None
) -> Optional[Path]:
    """
    Ensure model is available, downloading if necessary
    
    Priority:
    1. Local file (if exists)
    2. Hugging Face Hub
    3. Google Drive
    4. Direct URL
    
    Args:
        model_path: Local path to model
        model_url: Direct download URL
        gdrive_id: Google Drive file ID
        hf_repo: Hugging Face repo ID
        hf_filename: Filename in HF repo
        
    Returns:
        Path to model file or None if failed
    """
    model_path = Path(model_path)
    
    # Check if model already exists
    if model_path.exists():
        st.success(f"✅ Model found at {model_path}")
        return model_path
    
    st.warning(f"⚠️ Model not found at {model_path}, attempting download...")
    
    # Try Hugging Face first (fastest and most reliable)
    if hf_repo and hf_filename:
        st.info(f"📦 Trying Hugging Face: {hf_repo}")
        if download_from_huggingface(hf_repo, hf_filename, model_path):
            return model_path
    
    # Try Google Drive
    if gdrive_id:
        st.info(f"📦 Trying Google Drive")
        if download_from_google_drive(gdrive_id, model_path):
            return model_path
    
    # Try direct URL
    if model_url:
        st.info(f"📦 Trying direct download")
        if download_from_url(model_url, model_path):
            return model_path
    
    # All methods failed
    st.error("""
    ❌ Could not download model. Please:
    1. Upload model file to GitHub repository (< 100MB)
    2. Use Git LFS for larger files
    3. Provide valid download URL in Streamlit secrets
    4. Host on Hugging Face Hub (recommended)
    """)
    return None


def get_model_path_from_secrets() -> str:
    """Get model path from Streamlit secrets or environment"""
    try:
        # Try Streamlit secrets first
        if hasattr(st, 'secrets'):
            return st.secrets.get("model", {}).get("MODEL_PATH", "models/demo_run/best_model.pth")
    except:
        pass
    
    # Fall back to environment variable
    return os.getenv("MODEL_PATH", "models/demo_run/best_model.pth")


def get_download_config():
    """Get download configuration from secrets"""
    config = {
        "model_url": None,
        "gdrive_id": None,
        "hf_repo": None,
        "hf_filename": None
    }
    
    try:
        if hasattr(st, 'secrets') and 'model' in st.secrets:
            secrets = st.secrets['model']
            config['model_url'] = secrets.get('MODEL_URL')
            config['gdrive_id'] = secrets.get('GDRIVE_FILE_ID')
            config['hf_repo'] = secrets.get('HF_REPO_ID')
            config['hf_filename'] = secrets.get('HF_FILENAME')
    except:
        pass
    
    # Also check environment variables
    config['model_url'] = config['model_url'] or os.getenv('MODEL_URL')
    config['gdrive_id'] = config['gdrive_id'] or os.getenv('GDRIVE_FILE_ID')
    config['hf_repo'] = config['hf_repo'] or os.getenv('HF_REPO_ID')
    config['hf_filename'] = config['hf_filename'] or os.getenv('HF_FILENAME')
    
    return config


# Example usage in Streamlit app:
if __name__ == "__main__":
    st.title("Model Download Test")
    
    model_path = get_model_path_from_secrets()
    download_config = get_download_config()
    
    st.write(f"Model path: {model_path}")
    st.write(f"Download config: {download_config}")
    
    final_path = ensure_model_available(
        model_path=model_path,
        **download_config
    )
    
    if final_path:
        st.success(f"Model ready at: {final_path}")
        st.write(f"File size: {final_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        st.error("Model not available")
