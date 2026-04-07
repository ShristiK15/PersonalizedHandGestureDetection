import torch
from torch.utils.data import DataLoader
from dataset_loader import JesterSkeletonDataset
from validate import validate
from ddnet_model import DDNet
from config import device,batch_size


def main():
    test_dataset = JesterSkeletonDataset(
        "test_split.csv",
        r"C:\jester_skeletons_clean_01"
        # "csv_files/test_split.csv",
        # r"C:\jester_skeletons_clean_output"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print("Device:", device)
    print("Dataset size:", len(test_dataset))

    model = DDNet(num_classes=25).to(device)

    model.load_state_dict(torch.load("weights/ddnet_pretrained.pth"))
    # model.load_state_dict(torch.load("weights/ddnet_pretrained.pth"))

# 
    test_acc = validate(model, test_loader, device)

    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
