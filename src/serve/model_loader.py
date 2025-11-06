"""
Helper to resolve model file and class names paths for the Streamlit app.
This centralizes model discovery and optional download behavior so the UI code
is simpler and more robust.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Default search locations (relative to repo root)
FALLBACK_MODELS = [
    "models/streamlit_run/best_model.pth",
    "models/demo_run/best_model.pth",
    "models/best_model.pth",
]

def _try_import_downloader():
    """Attempt to import optional model downloader helpers from src.utils.
    Returns a tuple of (ensure_model_available_fn, get_model_path_from_secrets_fn, get_download_config_fn)
    or (None, None, None) if not available.
    """
    try:
        from src.utils.model_downloader import (
            ensure_model_available,
            get_model_path_from_secrets,
            get_download_config,
        )
        return ensure_model_available, get_model_path_from_secrets, get_download_config
    except Exception:
        return None, None, None

def resolve_model_and_classnames(streamlit_module=None) -> Tuple[Optional[str], Optional[str]]:
    """Resolve model path and class_names.txt path.

    Args:
        streamlit_module: Optional streamlit module object (pass `st` from the app).
                         This is used only to read secrets in Streamlit Cloud.

    Returns:
        A tuple (model_path:str or None, class_names_path:str or None)
    """
    # 1) Secrets (Streamlit Cloud/local secrets)
    secret_model_path = None
    if streamlit_module is not None:
        try:
            if hasattr(streamlit_module, "secrets") and "model" in streamlit_module.secrets:
                secret_model_path = streamlit_module.secrets["model"].get("MODEL_PATH")
        except Exception:
            secret_model_path = None

    # 2) Environment variables
    env_model_path = os.environ.get("MODEL_PATH", "").strip()
    env_model_dir = os.environ.get("MODEL_DIR", "").strip()

    # Build candidate list
    candidates = [env_model_path, secret_model_path] + FALLBACK_MODELS
    candidates = [c for c in candidates if c]

    chosen_model = None
    for c in candidates:
        p = Path(c)
        # Make relative to project root if not absolute
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists():
            chosen_model = str(p)
            break

    # If still not found, optionally try downloading via helper in src.utils.model_downloader
    ensure_model_available, get_model_path_from_secrets, get_download_config = _try_import_downloader()
    if chosen_model is None and ensure_model_available and get_model_path_from_secrets:
        try:
            target_path = get_model_path_from_secrets()
            cfg = get_download_config()
            resolved = ensure_model_available(target_path or FALLBACK_MODELS[0], **cfg)
            if resolved and resolved.exists():
                chosen_model = str(resolved)
        except Exception:
            logger.exception("Model download helper failed")

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