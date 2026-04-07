import os
import numpy as np
import random

ROOT = "DHG2016"
TARGET_FRAMES = 32

# =========================
# PREPROCESS FUNCTIONS
# =========================

def read_skeleton(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        values = list(map(float, line.strip().split()))
        joints = np.array(values).reshape(-1, 3)  # (22,3)
        data.append(joints)

    return np.array(data)  # (T,22,3)


# def resample(sequence, target_frames=32):
#     T, V, C = sequence.shape
#     new_seq = np.zeros((target_frames, V, C))

#     for i in range(V):
#         for j in range(C):
#             new_seq[:, i, j] = np.interp(
#                 np.linspace(0, T-1, target_frames),
#                 np.arange(T),
#                 sequence[:, i, j]
#             )
#     return new_seq

# =========================
# UPGRADED RESAMPLE FUNCTION
# =========================
def resample(sequence, target_frames=32):
    # Vectorized interpolation for massive speedup
    T, V, C = sequence.shape
    
    # Create the old and new time steps
    old_time = np.arange(T)
    new_time = np.linspace(0, T - 1, target_frames)
    
    # Reshape sequence to (T, V*C) so we can interpolate all coordinates at once
    seq_flat = sequence.reshape(T, -1)
    
    new_seq_flat = np.zeros((target_frames, V * C))
    for i in range(V * C):
        new_seq_flat[:, i] = np.interp(new_time, old_time, seq_flat[:, i])
        
    # Reshape back to (target_frames, V, C)
    return new_seq_flat.reshape(target_frames, V, C)


def normalize(sequence):
    origin = sequence[:, 0:1, :]
    return sequence - origin


# =========================
# LOAD + PROCESS DATA
# =========================

data, labels, users = [], [], []

for gesture in os.listdir(ROOT):
    
    # skip invalid folders
    if not gesture.startswith("gesture_"):
        continue

    parts = gesture.split("_")

    if len(parts) != 2 or not parts[1].isdigit():
        continue

    g_id = int(parts[1]) - 1
    gesture_path = os.path.join(ROOT, gesture)

    for finger in os.listdir(gesture_path):
        if not finger.startswith("finger_"):
            continue

        parts = finger.split("_")

        if len(parts) != 2 or not parts[1].isdigit():
            continue

        f_id = int(parts[1]) - 1
        finger_path = os.path.join(gesture_path, finger)
        label = g_id * 2 + f_id  # 28 classes

        for subject in os.listdir(finger_path):
            if not subject.startswith("subject_"):
                continue

            parts = subject.split("_")

            if len(parts) != 2 or not parts[1].isdigit():
                continue

            u_id = int(parts[1]) - 1
            subject_path = os.path.join(finger_path, subject)

            for essai in os.listdir(subject_path):
                file_path = os.path.join(subject_path, essai, "skeleton_world.txt")

                if not os.path.exists(file_path):
                    continue

                seq = read_skeleton(file_path)

                seq = seq[:, 1:, :]            # 22 → 21 joints
                seq = resample(seq, 32)        # fix frames
                seq = normalize(seq)           # center

                data.append(seq)
                labels.append(label)
                users.append(u_id)

data = np.array(data)
labels = np.array(labels)
users = np.array(users)

print("Processed Data:", data.shape)  # (2800,32,21,3)

def split_data(data, labels, users, class_list):
    mask = np.isin(labels, class_list)
    return data[mask], labels[mask], users[mask]

# =========================
# 10-SEED SPLIT & SAVE LOOP
# =========================

# We will generate 10 different splits using seeds 0 through 9
for current_seed in range(10):
    print(f"\n--- Generating Split for Seed {current_seed} ---")
    
    gesture_groups = list(range(14))  # 14 base gestures
    random.seed(current_seed)         # Apply the current seed
    random.shuffle(gesture_groups)

    train_g = gesture_groups[:7]
    remaining_g = gesture_groups[7:]

    # 1. Isolate 14 Training Classes
    meta_train_classes = []
    for g in train_g:
        meta_train_classes.extend([g * 2, g * 2 + 1])

    # 2. Extract remaining 14 classes
    remaining_classes = []
    for g in remaining_g:
        remaining_classes.extend([g * 2, g * 2 + 1])

    # Shuffle the remaining 14 classes
    random.shuffle(remaining_classes)

    # 3. Split 7 Validation / 7 Testing
    meta_val_classes = remaining_classes[:7]
    meta_test_classes = remaining_classes[7:]

    # Slice the data
    train_data, train_labels, train_users = split_data(data, labels, users, meta_train_classes)
    val_data, val_labels, val_users       = split_data(data, labels, users, meta_val_classes)
    test_data, test_labels, test_users    = split_data(data, labels, users, meta_test_classes)

    # =========================
    # CREATE DIRS & SAVE
    # =========================
    
    # Create a specific folder for this seed (e.g., dhg_clean/seed_0)
    seed_dir = f"dhg_clean/seed_{current_seed}"
    os.makedirs(seed_dir, exist_ok=True)

    np.save(f"{seed_dir}/train_data.npy", train_data)
    np.save(f"{seed_dir}/train_labels.npy", train_labels)
    np.save(f"{seed_dir}/train_users.npy", train_users)

    np.save(f"{seed_dir}/val_data.npy", val_data)
    np.save(f"{seed_dir}/val_labels.npy", val_labels)
    np.save(f"{seed_dir}/val_users.npy", val_users)

    np.save(f"{seed_dir}/test_data.npy", test_data)
    np.save(f"{seed_dir}/test_labels.npy", test_labels)
    np.save(f"{seed_dir}/test_users.npy", test_users)

print("\nAll 10 seeds successfully processed and saved! ✅")
# =========================
# META SPLIT (GESTURE-WISE)
# =========================

# gesture_groups = list(range(14))  # 14 base gestures
# random.seed(42)
# random.shuffle(gesture_groups)

# train_g = gesture_groups[:7]

# meta_train_classes = []
# for g in train_g:
#     meta_train_classes.extend([g * 2, g * 2 + 1])

# # 2. Take the remaining 7 groups (14 classes) for Val and Test
# remaining_g = gesture_groups[7:]

# remaining_classes = []
# for g in remaining_g:
#     remaining_classes.extend([g * 2, g * 2 + 1])

# # Shuffle the remaining 14 classes to split them evenly
# random.shuffle(remaining_classes)

# # 3. Split exactly 7 for Validation and 7 for Testing
# meta_val_classes = remaining_classes[:7]
# meta_test_classes = remaining_classes[7:]


# def split_data(data, labels, users, class_list):
#     mask = np.isin(labels, class_list)
#     return data[mask], labels[mask], users[mask]


# train_data, train_labels, train_users = split_data(data, labels, users, meta_train_classes)
# val_data, val_labels, val_users       = split_data(data, labels, users, meta_val_classes)
# test_data, test_labels, test_users    = split_data(data, labels, users, meta_test_classes)


# # =========================
# # CHECKS
# # =========================

# print("Train classes:", set(train_labels))
# print("Val classes:", set(val_labels))
# print("Test classes:", set(test_labels))

# print("Train shape:", train_data.shape)
# print("Val shape:", val_data.shape)
# print("Test shape:", test_data.shape)

# # ensure no overlap
# print("Overlap check:", set(train_labels) & set(test_labels))  # should be empty


# # =========================
# # SAVE
# # =========================

# np.save("dhg_clean/train_data.npy", train_data)
# np.save("dhg_clean/train_labels.npy", train_labels)
# np.save("dhg_clean/train_users.npy", train_users)

# np.save("dhg_clean/val_data.npy", val_data)
# np.save("dhg_clean/val_labels.npy", val_labels)
# np.save("dhg_clean/val_users.npy", val_users)

# np.save("dhg_clean/test_data.npy", test_data)
# np.save("dhg_clean/test_labels.npy", test_labels)
# np.save("dhg_clean/test_users.npy", test_users)

# print("Saved successfully ✅")