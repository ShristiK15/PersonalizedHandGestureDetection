import cv2
import mediapipe
from mediapipe.python.solutions import hands as mp_hands

import numpy as np
import os
import glob

# --- Configuration ---
print("code starts")
dataset_root = '20bn-jester-v1'
output_dir = 'jester_skeletons_output'

os.makedirs(output_dir, exist_ok=True)

# --- MediaPipe Setup ---
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def process_video_folder(video_id):

    video_path = os.path.join(dataset_root, video_id)

    frames = sorted(glob.glob(os.path.join(video_path, "*.jpg")))

    if not frames:
        return None

    skeleton_sequence = []

    for frame_path in frames:

        image = cv2.imread(frame_path)

        if image is None:
            skeleton_sequence.append(np.zeros((21,3), dtype=np.float32))
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            frame_landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32
            )

        else:

            frame_landmarks = np.zeros((21,3), dtype=np.float32)

        skeleton_sequence.append(frame_landmarks)

    return np.array(skeleton_sequence, dtype=np.float32)


def main():

    print(f"Scanning for video folders in: {dataset_root}")

    if not os.path.exists(dataset_root):

        print("Dataset folder not found")
        return

    all_video_ids = sorted(
        [f for f in os.listdir(dataset_root)
         if os.path.isdir(os.path.join(dataset_root, f))],
        key=lambda x: int(x)
    )

    total_videos = len(all_video_ids)

    print(f"Found {total_videos} videos")

    for index, video_id in enumerate(all_video_ids):

        save_path = os.path.join(output_dir, f"{video_id}.npy")

        if os.path.exists(save_path):
            continue

        try:

            sequence = process_video_folder(video_id)

            if sequence is not None:

                np.save(save_path, sequence)

            if index % 50 == 0:

                print(f"[{index}/{total_videos}] Processed {video_id}")

        except Exception as e:

            print(f"Error processing {video_id}: {e}")

    hands.close()

    print("Extraction complete")


if __name__ == "__main__":
    main()
