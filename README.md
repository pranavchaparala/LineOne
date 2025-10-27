# Voice Search Audio Snippets

A machine learning-powered search system that finds semantically similar audio snippets using AI embeddings.

## Features

- 🔍 **Semantic Search**: Uses sentence transformers to find relevant/related audio snippets
- 💬 **Text Query**: Enter text queries to find answers, continuations, or relevant content
- 🎤 **Voice Support**: Speech-to-text capability (when models are available)
- 🎵 **Smart Matching**: Finds snippets that are semantically relevant, not just word matches

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Backend

```bash
uvicorn backend:app --reload
```

The backend will:
- Load the Whisper speech-to-text model
- Load the sentence transformer model for semantic similarity
- Precompute embeddings for all snippets in `snippets.json` (this takes a few minutes on first run)

### 3. Open the Frontend

Open `index.html` in your web browser.

## How It Works

1. **Record Audio**: Click "Start Recording" and speak into your microphone
2. **Transcription**: Your speech is converted to text using Whisper
3. **Semantic Matching**: The system finds the most semantically similar snippet from your dataset
4. **Display Results**: Shows your transcription, the matched snippet, and similarity score

## Models Used

- **Speech-to-Text**: `openai/whisper-tiny` - Fast and accurate speech recognition
- **Semantic Similarity**: `all-MiniLM-L6-v2` - Efficient sentence embeddings for finding related content

## API Endpoints

- `GET /` - Health check
- `POST /search_voice` - Upload audio and get matched snippet
- `GET /audio_clips/{file_name}` - Serve audio files

## Customization

You can switch to different models in `backend.py`:

```python
# For better accuracy (slower):
processor = AutoProcessor.from_pretrained("openai/whisper-small")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")

# For faster inference (less accurate):
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = AutoModelForSpeechSeq2Seq.from_pretrained("facebook/wav2vec2-base-960h")
```

