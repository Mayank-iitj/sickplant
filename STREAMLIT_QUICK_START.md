# Streamlit Deployment Quick Reference

## 🚀 One-Command Deploy to Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [streamlit.io/cloud](https://streamlit.io/cloud))

### Deployment Steps

1. **Initialize Git & Push to GitHub**:
   ```bash
   # Initialize repository
   git init
   git add .
   git commit -m "Ready for Streamlit deployment"
   
   # Create GitHub repo and push
   git remote add origin https://github.com/YOUR_USERNAME/plant-disease-detector.git
   git branch -M main
   git push -u origin main
   ```

2. **Setup Git LFS for Model Files** (if model > 100MB):
   ```bash
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   git add models/demo_run/best_model.pth
   git commit -m "Add model with Git LFS"
   git push
   ```

3. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file: `streamlit_app.py`
   - Click "Deploy"

4. **Done!** Your app will be live at:
   ```
   https://YOUR_USERNAME-plant-disease-detector.streamlit.app
   ```

## 📦 Files Created

```
✓ streamlit_app.py              - Entry point for Streamlit Cloud
✓ requirements-streamlit.txt     - Optimized dependencies
✓ packages.txt                   - System packages (OpenCV, etc.)
✓ .streamlit/config.toml        - App configuration
✓ .streamlit/secrets.toml.example - Secrets template
✓ .gitattributes                - Git LFS config
✓ STREAMLIT_DEPLOYMENT.md       - Complete guide
```

## ⚡ Quick Test Locally

```bash
# Install Streamlit requirements
pip install -r requirements-streamlit.txt

# Run locally
streamlit run streamlit_app.py
```

Access at: http://localhost:8501

## 🔧 Configuration

### Update Theme (`.streamlit/config.toml`):
```toml
[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
```

### Set Secrets (Streamlit Cloud UI):
```toml
[model]
MODEL_PATH = "models/demo_run/best_model.pth"

[inference]
DEVICE = "cpu"
TOP_K = 5
```

## 🐛 Troubleshooting

### Model Not Found
- Ensure model is in repo (< 100MB) or use Git LFS
- Check `MODEL_PATH` in secrets matches actual path

### Import Errors
- Verify all dependencies in `requirements-streamlit.txt`
- Check `packages.txt` for system libraries

### Out of Memory
- Use smaller model (MobileNet)
- Enable model caching:
  ```python
  @st.cache_resource
  def load_model():
      return PlantDiseasePredictor(...)
  ```

### OpenCV Issues
- Ensure `packages.txt` exists with:
  ```
  libgl1-mesa-glx
  libglib2.0-0
  ```

## 📊 Streamlit Cloud Limits

- **Free Tier**:
  - 1 GB RAM
  - 1 CPU core
  - 1 GB storage
  - Public apps only

- **For More Resources**: Upgrade to Streamlit for Teams

## 🔗 Resources

- Full guide: `STREAMLIT_DEPLOYMENT.md`
- Streamlit Docs: https://docs.streamlit.io
- Community Forum: https://discuss.streamlit.io

## ✅ Pre-Deployment Checklist

- [ ] Code tested locally with `streamlit run streamlit_app.py`
- [ ] `requirements-streamlit.txt` includes all dependencies
- [ ] Model file is accessible (< 100MB or Git LFS configured)
- [ ] `.streamlit/config.toml` configured
- [ ] Secrets configured in Streamlit Cloud (if needed)
- [ ] All file paths are relative
- [ ] App tested with 1GB RAM limit

## 🎉 You're Ready!

Your Plant Disease Detector is now configured for Streamlit Cloud deployment. Follow the steps above and your app will be live in minutes!

**Next**: See `STREAMLIT_DEPLOYMENT.md` for advanced configuration, optimization tips, and production best practices.
