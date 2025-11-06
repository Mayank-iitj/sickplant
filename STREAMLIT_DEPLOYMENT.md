# Streamlit Cloud Deployment Guide

This guide explains how to deploy the Plant Disease Detector to Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub Account**: Your code must be in a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)
3. **Trained Model**: Have a trained model file ready

## 🚀 Quick Deployment Steps

### Step 1: Prepare Your Repository

1. **Push code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Streamlit deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/plant-disease-detector.git
   git push -u origin main
   ```

2. **Add model file** (choose one method):
   
   **Option A: Git LFS (for models < 100MB)**
   ```bash
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   git add models/demo_run/best_model.pth
   git commit -m "Add trained model"
   git push
   ```
   
   **Option B: External hosting (for larger models)**
   - Upload model to Google Drive, Dropbox, or Hugging Face
   - Get a direct download link
   - Update `MODEL_URL` in Streamlit secrets (Step 4)

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Fill in the form:
   - **Repository**: `YOUR_USERNAME/plant-disease-detector`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. Click **"Deploy"**

### Step 3: Configure Secrets (Optional)

If you need to customize settings:

1. In Streamlit Cloud, go to your app
2. Click **⚙️ Settings** → **Secrets**
3. Copy content from `.streamlit/secrets.toml.example`
4. Paste and modify as needed
5. Click **Save**

Example secrets:
```toml
[model]
MODEL_PATH = "models/demo_run/best_model.pth"
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"

[inference]
DEVICE = "cpu"
TOP_K = 5

[app]
MAX_UPLOAD_SIZE_MB = 50
ENABLE_GRADCAM = true
```

### Step 4: Handle Large Models

If your model is too large for GitHub (>100MB), use this approach:

1. **Upload model to external storage**:
   - Google Drive (get shareable link)
   - Hugging Face Hub
   - AWS S3 or similar

2. **Modify `src/serve/app_streamlit.py`** to download model on startup:
   ```python
   import requests
   from pathlib import Path
   
   def download_model(url: str, save_path: Path):
       if not save_path.exists():
           st.info("Downloading model... This may take a minute.")
           response = requests.get(url, stream=True)
           save_path.parent.mkdir(parents=True, exist_ok=True)
           with open(save_path, 'wb') as f:
               for chunk in response.iter_content(chunk_size=8192):
                   f.write(chunk)
           st.success("Model downloaded successfully!")
   
   # In your app initialization:
   model_url = st.secrets.get("model", {}).get("MODEL_URL", "")
   if model_url:
       download_model(model_url, Path("models/best_model.pth"))
   ```

3. **Add to requirements-streamlit.txt**:
   ```
   requests==2.31.0
   ```

## 📦 Files Created for Streamlit Deployment

```
plant_disease_detector/
├── streamlit_app.py              # Entry point for Streamlit Cloud
├── requirements-streamlit.txt     # Streamlit Cloud dependencies
├── packages.txt                   # System dependencies
├── .streamlit/
│   ├── config.toml               # Streamlit configuration
│   └── secrets.toml.example      # Secrets template
└── STREAMLIT_DEPLOYMENT.md       # This guide
```

## ⚙️ Configuration Files Explained

### 1. `streamlit_app.py`
- Entry point that Streamlit Cloud looks for
- Imports and runs the main app from `src/serve/app_streamlit.py`

### 2. `requirements-streamlit.txt`
- Optimized dependencies for Streamlit Cloud
- Uses CPU-only PyTorch to reduce size
- Includes all necessary ML libraries

### 3. `packages.txt`
- System-level dependencies (apt packages)
- Required for OpenCV and other libraries

### 4. `.streamlit/config.toml`
- UI theme and styling
- Server configuration
- Upload size limits

### 5. `.streamlit/secrets.toml`
- Private configuration (not committed to git)
- Model paths, API keys, etc.

## 🎨 Customization

### Change Theme Colors

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#4CAF50"      # Green for plant theme
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Increase Upload Limit

In `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200  # MB
```

### Add Custom Domain

1. In Streamlit Cloud: Settings → Custom domain
2. Enter your domain name
3. Add CNAME record to your DNS:
   ```
   CNAME: subdomain → your-app.streamlit.app
   ```

## 🔍 Troubleshooting

### Issue: App crashes on startup

**Solution**: Check logs in Streamlit Cloud
- Click "Manage app" → "Logs"
- Look for import errors or missing dependencies

### Issue: Model file not found

**Solutions**:
1. Verify model path in secrets
2. Check if Git LFS is properly configured
3. Use external model hosting (see Step 4)

### Issue: Out of memory

**Solutions**:
1. Use a smaller model (e.g., MobileNet instead of ResNet)
2. Reduce batch size in inference
3. Use model quantization:
   ```python
   model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

### Issue: Slow inference

**Solutions**:
1. Enable caching:
   ```python
   @st.cache_resource
   def load_model():
       return PlantDiseasePredictor(...)
   ```
2. Use ONNX for faster inference (see DEPLOYMENT.md)
3. Consider using Streamlit Community Cloud's compute tier

### Issue: OpenCV errors

**Solution**: Ensure `packages.txt` is present with required system libraries

### Issue: Dependencies take too long to install

**Solutions**:
1. Use `requirements-streamlit.txt` (optimized)
2. Remove unnecessary packages
3. Use pre-built wheels when possible

## 📊 Resource Limits (Streamlit Community Cloud)

- **CPU**: 1 core
- **Memory**: 1 GB RAM (800 MB available)
- **Storage**: 1 GB
- **Monthly usage**: 1 million requests or 1TB bandwidth

For higher limits, consider:
- Streamlit for Teams (paid)
- Self-hosting (Docker/Kubernetes)
- Other cloud platforms (see DEPLOYMENT.md)

## 🔒 Security Best Practices

1. **Never commit secrets**:
   - Add `.streamlit/secrets.toml` to `.gitignore`
   - Use Streamlit Cloud secrets manager

2. **Validate inputs**:
   - Check file types and sizes
   - Sanitize user uploads

3. **Rate limiting**:
   ```python
   from streamlit_autorefresh import st_autorefresh
   
   # Limit refresh rate
   count = st_autorefresh(interval=5000, limit=100)
   ```

4. **Add authentication** (optional):
   ```python
   import streamlit_authenticator as stauth
   
   authenticator = stauth.Authenticate(
       names, usernames, hashed_passwords,
       cookie_name, key, cookie_expiry_days
   )
   name, authentication_status, username = authenticator.login('Login', 'main')
   ```

## 🚀 Optimization Tips

### 1. Model Loading
```python
@st.cache_resource
def load_predictor(model_path: str):
    """Cache model to avoid reloading on every interaction"""
    return PlantDiseasePredictor(model_path=model_path, device="cpu")
```

### 2. Image Processing
```python
@st.cache_data
def process_image(image_bytes):
    """Cache processed images"""
    return Image.open(io.BytesIO(image_bytes))
```

### 3. Use Session State
```python
if 'predictor' not in st.session_state:
    st.session_state.predictor = load_predictor(MODEL_PATH)
```

### 4. Lazy Loading
```python
# Load heavy libraries only when needed
if st.button("Show Grad-CAM"):
    from core.explainability import GradCAM
    gradcam = GradCAM(model)
```

## 📈 Monitoring

### View Analytics
1. Go to Streamlit Cloud dashboard
2. Select your app
3. Click "Analytics" to see:
   - Active users
   - Response times
   - Error rates

### Add Custom Metrics
```python
import streamlit as st

# Track predictions
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0

st.session_state.prediction_count += 1
st.sidebar.metric("Total Predictions", st.session_state.prediction_count)
```

## 🔄 Continuous Deployment

Streamlit Cloud automatically redeploys when you push to GitHub:

```bash
# Make changes
git add .
git commit -m "Update model"
git push

# Streamlit Cloud will automatically detect and redeploy
```

### Disable Auto-Deploy
1. Go to app settings in Streamlit Cloud
2. Uncheck "Auto-deploy"
3. Manually trigger deploys when ready

## 🌐 Share Your App

After deployment, your app will be available at:
```
https://share.streamlit.io/YOUR_USERNAME/plant-disease-detector/main/streamlit_app.py
```

Or with custom domain:
```
https://your-custom-domain.com
```

Share the link with:
- QR code generator
- Social media
- Embed in website:
  ```html
  <iframe src="https://your-app.streamlit.app" width="800" height="600"></iframe>
  ```

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Forum](https://discuss.streamlit.io)
- [Example Apps Gallery](https://streamlit.io/gallery)

## 🆘 Support

If you encounter issues:

1. Check [Streamlit Community Forum](https://discuss.streamlit.io)
2. Review app logs in Streamlit Cloud
3. Test locally first: `streamlit run streamlit_app.py`
4. Open GitHub issue in your repository

## ✅ Deployment Checklist

Before deploying:

- [ ] Code is in GitHub repository
- [ ] `streamlit_app.py` is in root directory
- [ ] `requirements-streamlit.txt` is present and tested
- [ ] `packages.txt` includes system dependencies
- [ ] `.streamlit/config.toml` is configured
- [ ] Model file is accessible (Git LFS or external URL)
- [ ] Secrets are configured (if needed)
- [ ] App runs locally: `streamlit run streamlit_app.py`
- [ ] All imports work with requirements-streamlit.txt
- [ ] File paths are relative and cross-platform compatible
- [ ] Large files are handled appropriately (< 100MB)
- [ ] App is tested with limited resources (1GB RAM)

## 🎉 Success!

Your Plant Disease Detector is now live on Streamlit Cloud! 🌱

Users can access it 24/7 from any device with a browser. Monitor usage, gather feedback, and iterate based on real-world usage.

---

**Need help?** Check the troubleshooting section or reach out via GitHub issues.
