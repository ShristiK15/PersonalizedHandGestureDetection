import os
import numpy as np
from preprocess_skeleton import preprocess_skeleton

input_dir = "jester_skeletons_output"
output_dir = "jester_skeletons_clean_01"

os.makedirs(output_dir, exist_ok=True)

# ✅ Pre-filter only .npy files so index `i` is accurate
files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
print(f"Total .npy files found: {len(files)}")

for i, filename in enumerate(files):

    input_path = os.path.join(input_dir, filename)

    # Load
    try:
        sequence = np.load(input_path)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        continue

    # ✅ Validate shape
    if len(sequence.shape) != 3:
        print(f"⚠️ Skipping {filename}, invalid shape: {sequence.shape}")
        continue

    # Preprocess
    try:
        clean_sequence = preprocess_skeleton(sequence)
    except Exception as e:
        print(f"❌ Error preprocessing {filename}: {e}")
        continue

    # ✅ Preserve filename correctly (avoids double .npy)
    base_name = os.path.splitext(filename)[0]
    save_filename = base_name + ".npy"
    save_path = os.path.join(output_dir, save_filename)

    # Save
    try:
        np.save(save_path, clean_sequence)
    except Exception as e:
        print(f"❌ Error saving {save_filename}: {e}")
        continue

    # ✅ Debug first 10 files to verify naming
    if i < 10:
        print(f"DEBUG: {filename} → {save_filename}")

    # ✅ Accurate progress counter with total
    if (i + 1) % 100 == 0:
        print(f"✅ Processed {i + 1} / {len(files)} files")

print("🎉 All cleaned skeletons saved successfully.")