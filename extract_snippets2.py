from datasets import load_dataset
import os, json
from tqdm import tqdm
import soundfile as sf

# 1️⃣ Load the dataset
ds = load_dataset("SALT-NLP/spotify_podcast_ASR", split="train")

# 2️⃣ Create new output folder
os.makedirs("audio_clips_2", exist_ok=True)

data = []
START_INDEX = 100  # Skip first 100 since you already have them
MAX_ITEMS = 900    # 101 to 1000

# 3️⃣ Iterate and save
for i, example in tqdm(enumerate(ds), total=START_INDEX + MAX_ITEMS):
    if i < START_INDEX:
        continue
    if i >= START_INDEX + MAX_ITEMS:
        break

    transcript = example.get("transcription", "")
    filename = example.get("file_name", f"clip_{i}.wav")

    audio = example.get("audio")
    if audio is not None and "array" in audio:
        path = f"audio_clips_2/{i:04d}.wav"
        sf.write(path, audio["array"], audio["sampling_rate"])
    else:
        path = None

    data.append({
        "id": i,
        "file_name": filename,
        "audio": path,
        "transcription": transcript
    })

# 4️⃣ Save as JSON
with open("snippets_2.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Saved {len(data)} new clips (IDs {START_INDEX}–{START_INDEX + MAX_ITEMS - 1})")
