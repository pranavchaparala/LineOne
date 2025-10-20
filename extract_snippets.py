import os
from datasets import load_dataset, Dataset

# --- Configuration ---
CUSTOM_CACHE_DIR = "audio"
OUTPUT_FOLDER = "spotify_podcast_subset_100"

# --- Setup ---
os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True)
print(f"✅ Created/ensured cache directory: {os.path.abspath(CUSTOM_CACHE_DIR)}\n")

# --- Data Loading and Selection ---
print("Starting dataset load (metadata and index)...")
# 1. Load the entire 'train' split, directing the cache to the 'audio' folder
# num_proc=1 is critical to prevent segmentation faults on macOS during audio feature processing.
ds_split = load_dataset(
    "SALT-NLP/spotify_podcast_ASR",
    split="train",
    cache_dir=CUSTOM_CACHE_DIR,
    num_proc=1 
)
print("...Dataset metadata loaded.")

# 2. Select the first 100 examples
first_100_examples: Dataset = ds_split.select(range(100))
print(f"New subset size: {len(first_100_examples)} examples\n")

# 7. Force download of audio files
# Accessing the audio array for each example forces the download and caching of the file 
# into your 'audio' folder, now using the stable 'soundfile'/'librosa' backend.
print("⬇️ Forcing download of audio files for 100 examples (may take time)...")
for i in range(len(first_100_examples)):
    _ = first_100_examples[i]['audio']['array']

print("...Download complete! Audio files are now cached locally in the 'audio' folder.\n")

# 8. Save the Subset Snapshot (Optional, but recommended)
first_100_examples.save_to_disk(OUTPUT_FOLDER)
print(f"💾 Subset saved to local folder: {os.path.abspath(OUTPUT_FOLDER)}")

# --- Verification ---
print("\n--- First Example (Row 0) ---")
print(f"Text: {first_100_examples[0]['text']}")