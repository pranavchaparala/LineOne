from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

class QueryRequest(BaseModel):
    query: str

# ---------- Config ----------
SNIPPETS_JSON = "snippets.json"
TEMP_DIR = "temp"
AUDIO_FOLDER = "audio_clips"

# ---------- Load snippets ----------
print("Loading snippets...")
with open(SNIPPETS_JSON, "r") as f:
    snippets = json.load(f)
print(f"Loaded {len(snippets)} snippets!")

# ---------- Load ML Models ----------
print("Loading semantic similarity model (all-MiniLM-L6-v2)...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

# ---------- Precompute embeddings for all snippets ----------
print("Computing embeddings for all snippets (this may take a moment)...")
snippet_transcriptions = [s["transcription"] for s in snippets]
snippet_embeddings = sentence_model.encode(snippet_transcriptions, show_progress_bar=True, batch_size=32)
print(f"Computed {len(snippet_embeddings)} embeddings!")

# ---------- Initialize FastAPI ----------
app = FastAPI()

# Allow frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Ensure temp folder exists ----------
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------- Semantic search function ----------
def search_snippets(query_text):
    """
    Find semantically similar snippets using sentence transformers.
    This finds snippets that are answers, continuations, or relevant to the query.
    """
    print(f"Searching for: {query_text}")
    
    # Step 1: Compute embedding for query
    query_embedding = sentence_model.encode([query_text])
    
    # Step 2: Compute cosine similarity with all snippets
    # Normalize embeddings for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    snippet_norms = snippet_embeddings / np.linalg.norm(snippet_embeddings, axis=1, keepdims=True)
    
    similarities = np.dot(snippet_norms, query_norm.T).flatten()
    
    # Step 3: Find best match
    top_idx = int(np.argmax(similarities))
    top_score = float(similarities[top_idx])
    
    print(f"Best match score: {top_score:.4f}")
    print(f"Best match transcription: {snippets[top_idx]['transcription'][:100]}...")
    
    return snippets[top_idx], top_score

# ---------- API Endpoints ----------
@app.get("/")
async def root():
    return {"message": "Semantic search backend running! (Finds answers, continuations, and relevant snippets)"}

@app.post("/search_text")
async def search_text(request: QueryRequest):
    """
    Search snippets by text query using semantic similarity.
    Returns only the single best match.
    """
    # Search the dataset
    result, score = search_snippets(request.query)

    # Return query transcription, matched snippet, score, and audio file URL
    return {
        "query_transcription": request.query,
        "matched_transcription": result["transcription"],
        "audio_file": f"/audio_clips/{os.path.basename(result['audio'])}" if result["audio"] else None,
        "score": score
    }

@app.post("/search_voice")
async def search_voice(file: UploadFile = File(...)):
    """
    Placeholder for voice search (requires speech-to-text model).
    For now, returns a message to use text search instead.
    """
    return {
        "error": "Voice search not available yet. Please use text search at /search_text endpoint.",
        "message": "Upload audio files are not being processed. Use the text search interface instead."
    }

# ---------- Serve snippet audio files ----------
@app.get("/audio_clips/{file_name}")
async def serve_audio(file_name: str):
    path = os.path.join(AUDIO_FOLDER, file_name)
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/wav")
    return {"error": "File not found"}

