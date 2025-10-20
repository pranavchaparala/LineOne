from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import json
import numpy as np

# ---------- Config ----------
SNIPPETS_JSON = "snippets.json"  # your metadata file
TEMP_DIR = "temp"
AUDIO_FOLDER = "audio_clips"     # folder where snippet WAVs are stored

# ---------- Load snippets ----------
with open(SNIPPETS_JSON, "r") as f:
    snippets = json.load(f)

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

# ---------- Dummy search function ----------
def search_snippets(audio_path):
    """
    Replace this with your actual ML search logic.
    For now, it just picks a random snippet.
    """
    scores = np.random.rand(len(snippets))
    top_idx = int(np.argmax(scores))
    return snippets[top_idx], float(scores[top_idx])

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

    # Search the dataset
    result, score = search_snippets(audio_path)

    # Return transcription + matched score + audio file URL
    return {
        "transcription": result["transcription"],
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
