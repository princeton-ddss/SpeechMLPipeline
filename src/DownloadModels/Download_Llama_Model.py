from huggingface_hub import snapshot_download
from huggingface_hub import login
import os
# Log into huggingface
def download_llama_model(llama_model_local_dir, llama_model_repo_id='meta-llama/Llama-2-7b-chat-hf'):
    login(token='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')

    # Download LLaMA Model to Local Folder
    snapshot_download(repo_id = llama_model_repo_id,  cache_dir= llama_model_local_dir)


