# rag_pipeline.py

import os, json, numpy as np, faiss
from ollama import OllamaClient
from dotenv import load_dotenv

load_dotenv()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
ollama = OllamaClient(api_key=OLLAMA_API_KEY)

def build_index(chunks_dir: str, idx_path: str):
    vecs, ids = [], []
    for fn in sorted(os.listdir(chunks_dir), key=lambda x: int(x.split(".")[0])):
        data = json.loads(open(f"{chunks_dir}/{fn}").read())
        emb = ollama.embed(model="llama2-embedding", text=data["text"])
        vecs.append(np.array(emb, dtype="float32")); ids.append(data["id"])
    mat = np.vstack(vecs)
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat)
    faiss.write_index(idx, idx_path)
    print(f"[✓] Indexed {len(ids)} vectors")

if __name__ == "__main__":
    build_index("data/chunks", "squad_index.faiss")
