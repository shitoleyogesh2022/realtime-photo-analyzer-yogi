import os

# App settings
APP_TITLE = "📸 Mobile Photography Assistant"
APP_DESCRIPTION = "Get instant feedback on your photos!"

# File paths
UPLOAD_DIR = "uploads"
ANALYSIS_DIR = "analysis"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Model settings
IMAGE_SIZE = 224
BATCH_SIZE = 1
