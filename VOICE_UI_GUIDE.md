# 🎤 Voice UI Interface Guide

## What You Get

A beautiful iPhone-like interface that matches the image you shared, featuring:
- 📱 Realistic iPhone frame with Dynamic Island
- ✨ Smooth typing animations (character by character)
- 🎵 Automatic audio playback of matched snippets
- 💫 Subtle microinteractions (pulsing mic, glowing effects)
- 🎯 Single best match displayed as a continuous caption

## 🚀 How to Use

### 1. Make Sure Backend is Running

The backend should already be running. If not, start it:

```bash
cd /Users/pranavchaparala/Documents/Fall_2026/MoML/LineOne
source venv/bin/activate
uvicorn backend_semantic:app --reload
```

### 2. Open the Voice UI

Open `index_voice_ui.html` in your web browser:

```bash
open index_voice_ui.html
```

Or simply double-click the file.

### 3. Interact with the Interface

1. **Click the Microphone Button** at the bottom
2. **Watch the Magic:**
   - The mic turns blue and glows
   - "AI LISTENING..." appears with animated dots
   - Your query text types out character by character
   - The AI response types out as a continuation
   - Audio plays automatically when ready

## 🎨 Features

### Visual Design
- **Phone Frame**: Realistic iPhone-like frame with rounded corners
- **Dark Gradient**: Deep blue gradient background like in the image
- **Dynamic Island**: Black pill-shaped notch at the top
- **Status Bar**: Shows time, signal bars, and battery indicator
- **Home Indicator**: White bar at the bottom

### Microinteractions
- **Microphone Animation**: Pulses and glows when listening
- **Waveform Effect**: Expanding blue circles around the mic
- **Typing Animation**: Text appears word by word smoothly
- **Auto-scroll**: Transcript automatically scrolls as new text appears
- **Smooth Transitions**: All animations use easing functions

### Functionality
- **Semantic Search**: Uses AI to find the most relevant snippet
- **Single Best Match**: Shows only the top result (no list)
- **Continuous Caption**: Query and response flow together
- **Auto-play Audio**: Plays the matched audio automatically

## 🔧 Customization

### Change the Sample Query

Edit line 248 in `index_voice_ui.html`:

```javascript
const sampleQuery = "Your custom query here...";
```

### Adjust Typing Speed

Edit line 357 in `index_voice_ui.html`:

```javascript
}, 100); // Change this number (lower = faster)
```

### Change Colors

The phone frame and screen use CSS gradients. Edit the `.phone-frame` and `.screen` styles starting around line 15.

## 🎯 How It Works

1. **User clicks mic** → UI enters "listening" mode
2. **Query types out** → Simulated speech-to-text transcription
3. **Backend processes** → Semantic search finds best match
4. **Response types out** → AI response appears as continuation
5. **Audio plays** → Matched snippet audio plays automatically

## 🐛 Troubleshooting

**Backend not responding?**
```bash
# Check if backend is running
curl http://127.0.0.1:8000/

# Restart if needed
pkill -f "uvicorn backend_semantic"
uvicorn backend_semantic:app --reload
```

**Audio not playing?**
- Check browser console for errors
- Make sure audio files exist in `audio_clips/` folder
- Try a different browser (Chrome/Firefox recommended)

**Text not animating?**
- Check browser console for JavaScript errors
- Make sure you're using a modern browser
- Try refreshing the page

## 💡 Tips

- **Be Patient**: The typing animation takes time - it's intentional!
- **Try Different Queries**: Edit the `sampleQuery` variable to test different inputs
- **Check Console**: Open browser DevTools (F12) to see backend communication
- **Full Screen**: The phone frame looks best when centered on screen

## 🎨 Matching the Image

The interface was designed to match your reference image with:
- ✅ iPhone-like frame and screen
- ✅ Dark gradient background with blue glow
- ✅ Dynamic Island at the top
- ✅ Continuous text display (no separate sections)
- ✅ Microphone with pulsing animation
- ✅ "AI LISTENING..." indicator
- ✅ Automatic audio playback
- ✅ Subtle microinteractions throughout

Enjoy your new voice assistant interface! 🚀

