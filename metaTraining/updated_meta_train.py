import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from meta_ddnet_model import DDNet
from meta_dataset import FewShotDataset
from proto_net import proto_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CONFIG
# =========================
N_WAY = 5
K_SHOT = 1
K_QUERY = 15
EPISODES = 750
VAL_EPISODES = 75 # Defined by the paper

def train_meta():
    # Loop through all 10 seeds
    for current_seed in range(10):
        print(f"\n{'='*40}")
        print(f"🚀 STARTING META-TRAINING FOR SEED {current_seed}")
        print(f"{'='*40}")

        # 1. Dynamic Paths for the current seed
        data_dir = f"dhg_clean/seed_{current_seed}"
        
        train_dataset = FewShotDataset(
            f"{data_dir}/train_data.npy",
            f"{data_dir}/train_labels.npy",
            n_way=N_WAY, k_shot=K_SHOT, k_query=K_QUERY
        )
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        val_dataset = FewShotDataset(
            f"{data_dir}/val_data.npy",
            f"{data_dir}/val_labels.npy",
            n_way=N_WAY, k_shot=K_SHOT, k_query=K_QUERY
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

        # 2. Reset Model & Optimizer for this specific seed
        model = DDNet().to(DEVICE)
        # Always start from the base general Jester pretrained weights
        model.load_state_dict(torch.load("weights/ddnet_pretrained.pth"), strict=False)

        optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-3)

        # ADD THE SCHEDULER: Smoothly decays the learning rate to 0 over 750 episodes
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPISODES)

        best_val_acc = 0.0

        # =========================
        # META TRAIN LOOP
        # =========================
        for episode, batch in enumerate(train_loader):
            if episode >= EPISODES:
                break

            model.train()
            support_x, support_y, query_x, query_y = batch

            support_x = support_x.squeeze(0).to(DEVICE)
            support_y = support_y.squeeze(0).to(DEVICE)
            query_x   = query_x.squeeze(0).to(DEVICE)
            query_y   = query_y.squeeze(0).to(DEVICE)

            optimizer.zero_grad()
            loss, acc = proto_loss(model, support_x, support_y, query_x, query_y, N_WAY)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update the learning rate according to the schedule

            if episode % 50 == 0:
                print(f"Seed {current_seed} | Train Ep {episode} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")

            # =========================
            # META VALIDATION LOOP
            # =========================
            if episode > 0 and episode % 75 == 0:
                model.eval()
                total_val_acc = 0.0
                
                with torch.no_grad():
                    for val_ep, val_batch in enumerate(val_loader):
                        if val_ep >= VAL_EPISODES:
                            break
                            
                        v_sup_x, v_sup_y, v_que_x, v_que_y = val_batch
                        v_sup_x = v_sup_x.squeeze(0).to(DEVICE)
                        v_sup_y = v_sup_y.squeeze(0).to(DEVICE)
                        v_que_x = v_que_x.squeeze(0).to(DEVICE)
                        v_que_y = v_que_y.squeeze(0).to(DEVICE)

                        _, val_acc = proto_loss(model, v_sup_x, v_sup_y, v_que_x, v_que_y, N_WAY)
                        total_val_acc += val_acc.item()
                
                avg_val_acc = total_val_acc / VAL_EPISODES
                
                # Save dynamically using the seed number
                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    torch.save(model.state_dict(), f"meta_weights/meta_model_seed_{current_seed}.pth")
                    print(f"   [Val] Ep {episode}: Acc improved to {avg_val_acc:.4f} -> Saved ✅")

        print(f"Finished Seed {current_seed}. Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_meta()