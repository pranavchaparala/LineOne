import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# 1️⃣ Load your saved snippets (transcriptions)
with open("snippets.json", "r") as f:
    snippets = json.load(f)

texts = [s["transcription"] for s in snippets]
audio_files = [s["audio"] for s in snippets]

print(f"Loaded {len(texts)} snippets")

# 2️⃣ Load a text embedding model (this one is excellent for similarity)
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")


# 3️⃣ Encode all transcriptions once
embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

# 4️⃣ Interactive search
while True:
    query = input("\n🔍 Enter your search query (or 'exit'): ").strip()
    if query.lower() == "exit":
        break

    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, embeddings)[0]

    top_idx = int(scores.cpu().numpy().argmax())

    best_snippet = snippets[top_idx]

    print(f"\n🎧 Top match:")
    print(f"Text: {best_snippet['transcription']}")
    print(f"Audio file: {best_snippet['audio']} (score={float(scores[top_idx]):.3f})")
