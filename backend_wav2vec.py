from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import json
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
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
print("Loading speech-to-text model (Wav2Vec2)...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

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

# ---------- Ensure temp folder exists ----------
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------- Speech-to-Text function ----------
def transcribe_audio(audio_path):
    """
    Convert audio to text using Wav2Vec2 model.
    """
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Process audio
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Generate transcription
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# ---------- Semantic search function ----------
def search_snippets(audio_path):
    """
    Convert speech to text and find semantically similar snippets.
    """
    # Step 1: Convert audio to text
    query_text = transcribe_audio(audio_path)
    print(f"Transcribed query: {query_text}")
    
    # Step 2: Compute embedding for query
    query_embedding = sentence_model.encode([query_text])
    
    # Step 3: Compute cosine similarity with all snippets
    similarities = np.dot(snippet_embeddings, query_embedding.T).flatten()
    
    # Step 4: Find best match
    top_idx = int(np.argmax(similarities))
    top_score = float(similarities[top_idx])
    
    print(f"Best match score: {top_score:.4f}")
    print(f"Best match transcription: {snippets[top_idx]['transcription'][:100]}...")
    
    return snippets[top_idx], top_score, query_text

# ---------- API Endpoints ----------
@app.get("/")
async def root():
    return {"message": "Voice search backend running!"}

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

