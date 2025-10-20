import os
from datasets import load_dataset, Dataset

# --- Configuration ---
CUSTOM_CACHE_DIR = "audio"
OUTPUT_FOLDER = "spotify_podcast_subset_101_to_1000"

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

# 2. Select examples from index 100 (which is the 101st example) up to index 999 (which is the 1000th example).
# The total number of examples selected is 1000 - 100 = 900.
# The slice is range(start_index_inclusive, end_index_exclusive)
first_900_examples: Dataset = ds_split.select(range(100, 1000))
print(f"New subset size: {len(first_900_examples)} examples (101 to 1000)\n")

# 7. Force download of audio files
print("⬇️ Forcing download of audio files for 900 examples (this WILL take significant time)...")
for i in range(len(first_900_examples)):
    # Accessing the 'array' forces the library to fetch and store the audio data
    _ = first_900_examples[i]['audio']['array']

print("...Download complete! Audio files are now cached locally in the 'audio' folder.\n")

# 8. Save the Subset Snapshot (Optional, but recommended)
first_900_examples.save_to_disk(OUTPUT_FOLDER)
print(f"💾 Subset saved to local folder: {os.path.abspath(OUTPUT_FOLDER)}")

# --- Verification ---
print("\n--- First Example of this subset (Row 101) ---")
print(f"Index 100 (in the subset): {first_900_examples[0]['text']}")
print("\n--- Last Example of this subset (Row 1000) ---")
print(f"Index 899 (in the subset): {first_900_examples[-1]['text']}")