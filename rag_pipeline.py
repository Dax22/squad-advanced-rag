# rag_pipeline.py

import os
import json
import numpy as np
import faiss
from ollama import OllamaClient

# 1. Configure your Ollama client
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "YOUR_GROQCLOUD_KEY")
ollama = OllamaClient(api_key=OLLAMA_API_KEY)

def test_embedding():
    """
    Quick check: embed a sample string and print vector length.
    """
    sample = "Hello, world!"
    emb = ollama.embed(model="llama2-embedding", text=sample)
    print(f"Embedding length: {len(emb)}")

def build_faiss_index(chunks_dir: str, index_path: str):
    """
    Reads each chunk JSON, embeds its text, and builds a Faiss index.
    """
    # Gather embeddings and IDs
    vectors = []
    ids = []
    for fname in sorted(os.listdir(chunks_dir), key=lambda x: int(x.split(".")[0])):
        path = os.path.join(chunks_dir, fname)
        chunk = json.loads(open(path, "r", encoding="utf-8").read())
        text = chunk["text"]
        emb = ollama.embed(model="llama2-embedding", text=text)
        vectors.append(np.array(emb, dtype="float32"))
        ids.append(chunk["id"])

    # Stack into a single numpy array
    matrix = np.vstack(vectors)
    dim = matrix.shape[1]

    # Create a Faiss index (Inner-Product for cosine similarity)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss.write_index(index, index_path)

    print(f"[✓] Indexed {len(ids)} vectors → '{index_path}'")

if __name__ == "__main__":
    # 1. Test that embeddings work:
    print("→ Testing embedding:")
    test_embedding()

    # 2. Build the Faiss index from your chunk files
    print("\n→ Building Faiss index:")
    build_faiss_index("data/chunks", "squad_index.faiss")
