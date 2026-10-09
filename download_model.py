import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
    filename="qwen2.5-3b-instruct-q4_k_m.gguf",
    local_dir="models/",
)
print(f"Модель скачана: {model_path}")