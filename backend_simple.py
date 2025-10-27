from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import json
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import re
import warnings
warnings.filterwarnings("ignore")

# ---------- Config ----------
SNIPPETS_JSON = "snippets.json"
TEMP_DIR = "temp"
AUDIO_FOLDER = "audio_clips"

# Common words to ignore (stop words)
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'up', 'about', 'into', 'through', 'during', 'including', 'against', 'among',
    'throughout', 'despite', 'towards', 'upon', 'concerning', 'to', 'of', 'in', 'for', 'on',
    'at', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'including', 'against',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'will', 'would', 'should', 'could', 'may', 'might', 'must',
    'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose', 'where',
    'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn',
    'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan',
    'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'get', 'got', 'getting', 'got', 'gets',
    'go', 'goes', 'going', 'went', 'gone', 'come', 'comes', 'coming', 'came', 'see', 'saw',
    'seen', 'seeing', 'know', 'knew', 'known', 'knowing', 'think', 'thought', 'thinking',
    'take', 'took', 'taken', 'taking', 'give', 'gave', 'given', 'giving', 'make', 'made',
    'making', 'say', 'said', 'saying', 'tell', 'told', 'telling', 'want', 'wanted', 'wanting',
    'use', 'used', 'using', 'find', 'found', 'finding', 'try', 'tried', 'trying', 'work',
    'worked', 'working', 'call', 'called', 'calling', 'ask', 'asked', 'asking', 'need',
    'needed', 'needing', 'feel', 'felt', 'feeling', 'become', 'became', 'becoming',
    'leave', 'left', 'leaving', 'put', 'putting', 'mean', 'meant', 'meaning', 'keep',
    'kept', 'keeping', 'let', 'letting', 'begin', 'began', 'begun', 'beginning', 'seem',
    'seemed', 'seeming', 'help', 'helped', 'helping', 'talk', 'talked', 'talking', 'turn',
    'turned', 'turning', 'start', 'started', 'starting', 'show', 'showed', 'shown', 'showing',
    'hear', 'heard', 'hearing', 'play', 'played', 'playing', 'run', 'ran', 'running', 'move',
    'moved', 'moving', 'live', 'lived', 'living', 'believe', 'believed', 'believing',
    'bring', 'brought', 'bringing', 'happen', 'happened', 'happening', 'write', 'wrote',
    'written', 'writing', 'sit', 'sat', 'sitting', 'stand', 'stood', 'standing', 'lose',
    'lost', 'losing', 'pay', 'paid', 'paying', 'meet', 'met', 'meeting', 'include',
    'included', 'including', 'continue', 'continued', 'continuing', 'set', 'setting', 'learn',
    'learned', 'learning', 'change', 'changed', 'changing', 'lead', 'led', 'leading', 'understand',
    'understood', 'understanding', 'watch', 'watched', 'watching', 'follow', 'followed',
    'following', 'stop', 'stopped', 'stopping', 'create', 'created', 'creating', 'speak',
    'spoke', 'spoken', 'speaking', 'read', 'reading', 'allow', 'allowed', 'allowing',
    'add', 'added', 'adding', 'spend', 'spent', 'spending', 'grow', 'grew', 'grown', 'growing',
    'open', 'opened', 'opening', 'walk', 'walked', 'walking', 'win', 'won', 'winning',
    'offer', 'offered', 'offering', 'remember', 'remembered', 'remembering', 'love', 'loved',
    'loving', 'consider', 'considered', 'considering', 'appear', 'appeared', 'appearing',
    'buy', 'bought', 'buying', 'wait', 'waited', 'waiting', 'serve', 'served', 'serving',
    'die', 'died', 'dying', 'send', 'sent', 'sending', 'build', 'built', 'building',
    'stay', 'stayed', 'staying', 'fall', 'fell', 'fallen', 'falling', 'cut', 'cutting',
    'reach', 'reached', 'reaching', 'kill', 'killed', 'killing', 'raise', 'raised', 'raising'
}

# ---------- Load snippets ----------
print("Loading snippets...")
with open(SNIPPETS_JSON, "r") as f:
    snippets = json.load(f)

# ---------- Load ML Models ----------
print("Loading speech-to-text model (Wav2Vec2)...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
print("Model loaded successfully!")

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

# ---------- Helper function to extract meaningful words ----------
def extract_words(text):
    """Extract words from text, remove stop words, and return lowercase set"""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split into words
    words = text.split()
    # Filter out stop words and short words
    meaningful_words = {w for w in words if w not in STOP_WORDS and len(w) > 2}
    return meaningful_words

# ---------- Speech-to-Text function ----------
def transcribe_audio(audio_path):
    """Convert audio to text using Wav2Vec2 model"""
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

# ---------- Word-based search function ----------
def search_snippets(audio_path):
    """
    Convert speech to text and find snippet with highest word overlap.
    """
    # Step 1: Convert audio to text
    query_text = transcribe_audio(audio_path)
    print(f"Transcribed query: {query_text}")
    
    # Step 2: Extract meaningful words from query
    query_words = extract_words(query_text)
    print(f"Query meaningful words: {query_words}")
    
    # Step 3: Find snippet with most matching words
    best_match_idx = 0
    best_match_score = 0
    best_matching_words = []
    
    for idx, snippet in enumerate(snippets):
        snippet_words = extract_words(snippet["transcription"])
        # Count overlapping words
        overlap = query_words & snippet_words
        match_score = len(overlap)
        
        if match_score > best_match_score:
            best_match_score = match_score
            best_match_idx = idx
            best_matching_words = list(overlap)
    
    print(f"Best match score: {best_match_score} words")
    print(f"Matched words: {best_matching_words}")
    print(f"Best match transcription: {snippets[best_match_idx]['transcription'][:100]}...")
    
    return snippets[best_match_idx], best_match_score, query_text

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

