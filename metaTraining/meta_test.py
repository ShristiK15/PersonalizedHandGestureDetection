import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup paths for local imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from meta_ddnet_model import DDNet
from meta_dataset import FewShotDataset
from proto_net import proto_loss

# =========================
#Freeze BatchNorm layers during fine-tuning to prevent overfitting on small support sets
# This is a common practice in few-shot learning to maintain stable feature distributions
# def set_bn_eval(m):
#     """Forces BatchNorm layers to stay in evaluation mode during training."""
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         m.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CONFIG FOR PERSONAL STRATEGY
# =========================
N_WAY = 4
K_SHOT = 1  # Note: The paper tests this for k=1, 2, 3, and 4
# Dynamically use the remaining samples for that user

IS_PERSONAL = False  # Toggle this to True/False to switch strategies

if IS_PERSONAL:
    K_QUERY = N_WAY - K_SHOT  # Math constraint from the paper (only 5 samples per user exist)
else:
    K_QUERY = 15          # General strategy uses 15 query samples per episode

FINE_TUNE_STEPS = 10
EPISODES = 150 # The paper uses ~150 episodes for evaluation

def test():
    print(f"Starting {N_WAY}-Way {K_SHOT}-Shot Personal Fine-Tuning Evaluation...")
    print("Evaluating across 10 random seeds...\n")
    
    grand_total_accuracy = 0.0

    # Loop through all 10 seeds
    for current_seed in range(10):
        data_dir = f"dhg_clean/seed_{current_seed}"
        
        # 1. Load the Test Dataset for this specific seed
        
        dataset = FewShotDataset(
            data_path=f"{data_dir}/test_data.npy",
            label_path=f"{data_dir}/test_labels.npy",
            user_path=f"{data_dir}/test_users.npy" if IS_PERSONAL else None,
            n_way=N_WAY,
            k_shot=K_SHOT,
            k_query=K_QUERY,
            is_personal=IS_PERSONAL
        )

        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        # 2. Load the specific Meta-Trained Model for this seed
        base_model = DDNet().to(DEVICE)
        base_model.load_state_dict(torch.load(f"weights/meta_model_seed_{current_seed}.pth"))
        base_model.eval() 
        
        seed_total_acc = 0.0

        # 3. Meta-Testing Loop for this seed
        for episode, batch in enumerate(loader):
            if episode >= EPISODES:
                break

            model = DDNet().to(DEVICE)
            model.load_state_dict(base_model.state_dict())

            optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

            support_x, support_y, query_x, query_y = batch
            support_x = support_x.squeeze(0).to(DEVICE)
            support_y = support_y.squeeze(0).to(DEVICE)
            query_x   = query_x.squeeze(0).to(DEVICE)
            query_y   = query_y.squeeze(0).to(DEVICE)

            # Fine-tune
            model.train()

            # THE FIX: Freeze Batch Norm running statistics to prevent small-batch corruption
            # model.apply(set_bn_eval)

            for _ in range(FINE_TUNE_STEPS):
                optimizer.zero_grad()
                loss, _ = proto_loss(model, support_x, support_y, support_x, support_y, N_WAY)
                loss.backward()
                optimizer.step()

            # Test
            model.eval()
            with torch.no_grad():
                _, acc = proto_loss(model, support_x, support_y, query_x, query_y, N_WAY)

            seed_total_acc += acc.item()

        # Calculate final accuracy for this single seed
        seed_final_acc = (seed_total_acc / EPISODES) * 100
        print(f"Seed {current_seed} Accuracy: {seed_final_acc:.2f}%")
        
        grand_total_accuracy += seed_final_acc

    # Calculate the definitive final accuracy across all 10 seeds
    final_10_seed_average = grand_total_accuracy / 10
    print(f"\n{'='*50}")
    print(f"🏆 FINAL {N_WAY}-WAY {K_SHOT}-SHOT 10-SEED AVERAGE: {final_10_seed_average:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    test()