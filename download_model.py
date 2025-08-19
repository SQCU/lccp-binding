### 3. `download_model.py`
#This helper script fetches a model so you don't have to do it manually.
# download_model.py
from huggingface_hub import hf_hub_download
import os

# Ensure the 'models' directory exists
os.makedirs("models", exist_ok=True)

# Define the model to download
# Using a small, low-quality model for the example.
# You can replace this with any GGUF model from the Hub.
MODEL_REPO = "skymizer/gemma-3-1b-pt-qat-q4_0-gguf"
MODEL_FILE = "gemma-3-1b-pt-q4_0.gguf"

# Check if the model already exists
model_path = os.path.join("models", MODEL_FILE)
if os.path.exists(model_path):
    print(f"Model '{MODEL_FILE}' already exists in the 'models' directory. Skipping download.")
else:
    print(f"Downloading model '{MODEL_FILE}' from '{MODEL_REPO}'...")
    hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir="models",
        local_dir_use_symlinks=False
    )
    print("Model downloaded successfully.")