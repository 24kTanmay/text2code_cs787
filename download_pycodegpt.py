from huggingface_hub import hf_hub_download
import os

REPO = "Daoguang/PyCodeGPT"


TARGET_DIR = os.path.expanduser("~/GenAi/text2code_mrpt/models/pycodegpt-100m")
os.makedirs(TARGET_DIR, exist_ok=True)

files = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
#   "added_tokens.json",
    "pytorch_model.bin",
#   "pytorch_model.safetensors",
#   "tf_model.h5",
]

print(f"Downloading from repo: {REPO}\nTarget dir: {TARGET_DIR}\n")

for fname in files:
    try:
        path = hf_hub_download(repo_id=REPO, filename=fname, local_dir=TARGET_DIR, local_dir_use_symlinks=False)
        print("Downloaded", fname, "->", path)
    except Exception as e:
        print("Could not download", fname, ":", e)

print("\nDone. Files in target:", os.listdir(TARGET_DIR))
