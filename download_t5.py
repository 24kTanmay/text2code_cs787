from huggingface_hub import snapshot_download
import os

# ======================= PROXY SETTINGS =======================
# If you are behind a proxy, UNCOMMENT the two lines below
# and replace with your proxy address.
#
# os.environ["http_proxy"] = "http://your.proxy.address:port"
# os.environ["https_proxy"] = "http://your.proxy.address:port"
#
# If your proxy needs a username and password:
# os.environ["http_proxy"] = "http://username:password@your.proxy.address:port"
# os.environ["https_proxy"] = "http://username:password@your.proxy.address:port"
# =============================================================

# --- NEW, PUBLICLY AVAILABLE MODEL ---
model_name = "ramsrigouthamg/t5_paraphraser"
local_dir = "./t5_paraphraser_model"

# Make sure the target directory exists
os.makedirs(local_dir, exist_ok=True)

print(f"Downloading model '{model_name}' to '{local_dir}'...")

try:
    # This command downloads all files from the repo
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print("Download complete!")
    print(f"Files saved in: {os.path.abspath(local_dir)}")

except Exception as e:
    print(f"FATAL: Failed to download the model.")
    print(f"Please check your proxy settings, internet connection, or firewall.")
    print(f"Error: {e}")
