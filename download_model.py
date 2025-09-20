# download_model.py
from langchain_community.embeddings import HuggingFaceEmbeddings

print("Downloading and caching embedding model...")
HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Model download complete.")
