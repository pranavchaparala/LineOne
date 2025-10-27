# Semantic Search Setup Guide

## 🚀 Quick Start

### 1. Start the Backend
```bash
cd /Users/pranavchaparala/Documents/Fall_2026/MoML/LineOne
source venv/bin/activate
uvicorn backend_semantic:app --reload
```

### 2. Open the Frontend
Open `index_semantic.html` in your web browser.

## 🎯 What It Does

The semantic search system uses **AI-powered embeddings** to find snippets that are:
- **Answers** to your query
- **Continuations** of your topic
- **Semantically relevant** even without exact word matches

### Example Queries That Work Well:
- "I want to talk about music and entertainment"
- "What do you think about politics?"
- "Tell me about your podcast ideas"
- "How do you feel about making money?"
- "What's your goal with this podcast?"

## 📊 How It Works

1. **Sentence Transformer Model**: Uses `all-MiniLM-L6-v2` to convert text into semantic embeddings
2. **Precomputed Embeddings**: All 1000 snippets are embedded on startup
3. **Cosine Similarity**: Finds the most semantically similar snippets
4. **Top 5 Results**: Shows the best match plus 4 other relevant options

## 🔧 Available Backends

| Backend | Method | Use Case |
|---------|--------|----------|
| `backend_semantic.py` | **Semantic similarity** (AI embeddings) | Find answers, continuations, relevant content |
| `backend_wordmatch.py` | Word matching | Find exact word overlaps |
| `backend_simple.py` | Wav2Vec2 + word matching | Voice input with word matching |
| `backend.py` | Whisper + semantic | Voice input with semantic search (best accuracy) |

## 🎨 Frontend Options

- `index_semantic.html` - Semantic search with top 5 results (recommended)
- `index_text.html` - Simple text search with word matching
- `index.html` - Voice recording interface

## 📈 Performance

- **Startup time**: ~30 seconds (downloads model + computes embeddings)
- **Search time**: <100ms per query
- **Accuracy**: Finds semantically relevant content even without exact word matches

## 🎤 Adding Voice Support

To add speech-to-text, you'll need to:

1. Wait for Hugging Face servers to stabilize
2. Use `backend_simple.py` (Wav2Vec2) or `backend.py` (Whisper)
3. The voice recording interface in `index.html` will work automatically

## 🐛 Troubleshooting

**Backend won't start?**
```bash
# Kill any existing servers
pkill -f "uvicorn backend"

# Start fresh
uvicorn backend_semantic:app --reload
```

**Model download fails?**
- Hugging Face servers may be down
- Try again in a few minutes
- Or use `backend_wordmatch.py` as a fallback

**Port 8000 already in use?**
```bash
# Kill the process using port 8000
lsof -ti:8000 | xargs kill -9
```

## ✨ Tips

1. **Be conversational**: The semantic search works best with natural language
2. **Ask questions**: Try "What do you think about..." or "Tell me about..."
3. **Use context**: Longer queries often get better matches
4. **Check multiple results**: The top 5 matches give you variety

