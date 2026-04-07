import os
import numpy as np
import pandas as pd

def save_sample(sequence, gesture_name, sample_id):

    folder = f"my_gesture_dataset/{gesture_name}"

    os.makedirs(folder, exist_ok=True)

    path = f"{folder}/{sample_id}.npy"

    np.save(path, sequence)

    return path


def update_labels(file_path, gesture_name):

    csv_path = "my_gesture_dataset/labels.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["file","label"])

    df.loc[len(df)] = [file_path, gesture_name]

    df.to_csv(csv_path,index=False)