"""
Streamlit Cloud Entry Point
This is the main entry point for Streamlit Cloud deployment.
"""
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the main Streamlit app
from src.serve.app_streamlit import main

if __name__ == "__main__":
    main()
