import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------
# CONFIG
# -------------------------

CSV_PATH = "splittingTrainTestData\All_classes.csv"
# DATA_DIR = "dataset"
DATA_DIR = "jester_skeletons_clean_01"
# DATA_DIR = "jester_skeletons_clean_output"
FILE_EXTENSION = ".npy"

REMOVE_CLASSES = ["No gesture", "Doing other things"]

# -------------------------
# LOAD CSV (semicolon format)
# -------------------------

df = pd.read_csv(CSV_PATH, sep=";", header=None)
df.columns = ["video_id", "class"]

print("Original dataset size:", len(df))

# -------------------------
# REMOVE UNWANTED CLASSES
# -------------------------

df = df[~df["class"].str.lower().isin(["no gesture", "doing other things"])]

print("After removing unwanted classes:", len(df))

# -------------------------
# REMOVE MISSING FILES
# -------------------------

def file_exists(video_id):
    path = os.path.join(DATA_DIR, f"{video_id}{FILE_EXTENSION}")
    return os.path.exists(path)

df = df[df["video_id"].apply(file_exists)]

print("After removing missing files:", len(df))

# -------------------------
# ENCODE LABELS
# -------------------------

le = LabelEncoder()
df["label"] = le.fit_transform(df["class"])

print("\nClasses:")
print(le.classes_)

# -------------------------
# SPLIT DATASET
# -------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

print("\nDataset split:")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

# -------------------------
# SAVE SPLITS
# -------------------------

train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)

# train_df.to_csv("csv_files/train_split.csv", index=False)
# val_df.to_csv("csv_files/val_split.csv", index=False)
# test_df.to_csv("csv_files/test_split.csv", index=False)

print("\nSplits saved successfully!")