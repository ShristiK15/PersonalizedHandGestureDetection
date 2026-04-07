from torch.utils.data import DataLoader
from dataset_loader import JesterSkeletonDataset
from config import batch_size

def get_dataloaders():  
    train_dataset = JesterSkeletonDataset(
        "train_split.csv",
        "C:\jester_skeletons_clean_01"
        # "csv_files/train_split.csv",
        # "C:\jester_skeletons_clean_output"
    )

    val_dataset = JesterSkeletonDataset(
        "val_split.csv",
        "C:\jester_skeletons_clean_01"
        # "csv_files/val_split.csv",
        # "C:\jester_skeletons_clean_output"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader