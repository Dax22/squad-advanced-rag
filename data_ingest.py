# data_ingest.py

import json, os
from pathlib import Path

def split_to_chunks(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_path) as f:
        squad = json.load(f)
    count = 0
    for art in squad.get("data", []):
        for para in art.get("paragraphs", []):
            text = para.get("context", "").strip()
            if not text: continue
            Path(f"{output_dir}/{count}.json").write_text(
                json.dumps({"id": count, "text": text}, ensure_ascii=False)
            )
            count += 1
    print(f"[✓] Wrote {count} chunks to {output_dir}")

if __name__ == "__main__":
    split_to_chunks("data/train-v2.0.json", "data/chunks")

