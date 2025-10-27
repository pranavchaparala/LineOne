from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel # ADDED: Need this for the search_text endpoint
import os
import json
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from sentence_transformers import SentenceTransformer
import librosa
import warnings
warnings.filterwarnings("ignore")

# ---------- Config ----------
SNIPPETS_JSON = "snippets.json"  # your metadata file
TEMP_DIR = "temp"
AUDIO_FOLDER = "audio_clips"     # folder where snippet WAVs are stored

# ---------- Load snippets ----------
print("Loading snippets...")
with open(SNIPPETS_JSON, "r") as f:
    snippets = json.load(f)

# ---------- Load ML Models ----------
print("Loading speech-to-text model (Whisper)...")
processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")

print("Loading semantic similarity model...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Precompute embeddings for all snippets ----------
print("Computing embeddings for all snippets (this may take a moment)...")
snippet_transcriptions = [s["transcription"] for s in snippets]
snippet_embeddings = sentence_model.encode(snippet_transcriptions, show_progress_bar=True)
print(f"Computed {len(snippet_embeddings)} embeddings!")

# ---------- Initialize FastAPI ----------
app = FastAPI()

# Allow frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic Model for Text Input ----------
class Query(BaseModel):
    query: str
# ---------------------------------------------------

# ---------- Ensure temp folder exists ----------
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------- Speech-to-Text function (Unchanged) ----------
def transcribe_audio(audio_path):
    """
    Convert audio to text using Whisper model.
    """
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Process audio
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    
    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"])
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# ---------- Core Semantic search logic function (Modified to take text or audio) ----------
def search_snippets_by_text(query_text):
    """
    Find semantically similar snippets based on a text query.
    """
    # Step 1: Compute embedding for query
    query_embedding = sentence_model.encode([query_text])
    
    # Step 2: Compute cosine similarity with all snippets
    similarities = np.dot(snippet_embeddings, query_embedding.T).flatten()
    
    # Step 3: Find best match
    top_idx = int(np.argmax(similarities))
    top_score = float(similarities[top_idx])
    
    print(f"Text Query: {query_text}")
    print(f"Best match score: {top_score:.4f}")
    print(f"Best match transcription: {snippets[top_idx]['transcription'][:100]}...")
    
    return snippets[top_idx], top_score

# ---------- Modified search_snippets to use the new core logic ----------
def search_snippets(audio_path):
    # Step 1: Convert audio to text
    query_text = transcribe_audio(audio_path)
    
    # Step 2 & 3: Find best match using the new core function
    result, score = search_snippets_by_text(query_text)

    return result, score, query_text


# ---------- API Endpoints ----------
@app.get("/")
async def root():
    return {"message": "Voice search backend running!"}

# NEW ENDPOINT: Handles text input from the HTML client
@app.post("/search_text")
async def search_text(query_data: Query):
    """
    Accepts a text query and returns the semantically best-matched audio clip.
    """
    # Search the dataset using the new text-based function
    result, score = search_snippets_by_text(query_data.query)

    # Return matched snippet, score, and audio file URL
    # The client expects 'query_transcription' and 'matched_transcription'
    return {
        "query_transcription": query_data.query,
        "matched_transcription": result["transcription"],
        "audio_file": f"/audio_clips/{os.path.basename(result['audio'])}" if result["audio"] else None,
        "score": score
    }
# ----------------------------------------------------------------------


@app.post("/search_voice")
async def search_voice(file: UploadFile = File(...)):
    # Save uploaded voice file
    audio_path = os.path.join(TEMP_DIR, file.filename)
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # Search the dataset (now returns query_text too)
    result, score, query_text = search_snippets(audio_path)

    # Return query transcription, matched snippet, score, and audio file URL
    return {
        "query_transcription": query_text,
        "matched_transcription": result["transcription"],
        "audio_file": f"/audio_clips/{os.path.basename(result['audio'])}" if result["audio"] else None,
        "score": score
    }

# ---------- Serve snippet audio files ----------
@app.get("/audio_clips/{file_name}")
async def serve_audio(file_name: str):
    path = os.path.join(AUDIO_FOLDER, file_name)
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/wav")
    return {"error": "File not found"}