import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup paths for local imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from meta_ddnet_model import DDNet
from meta_dataset import FewShotDataset
from proto_net import compute_prototypes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CONFIGURATION
# =========================
N_WAY = 5
K_SHOT = 1
IS_PERSONAL = False  # Toggle: True for 85% target, False for 68% target

if IS_PERSONAL:
    K_QUERY = 5 - K_SHOT # Strict math constraint from the paper
else:
    K_QUERY = 15 

FINE_TUNE_STEPS = 10
EPISODES = 150 

def set_bn_eval(m):
    """Prevents small-batch corruption of BatchNorm statistics during fine-tuning"""
    if isinstance(m, nn.BatchNorm1d):
        m.eval()

# =========================
# THE EXPLICIT CLASSIFIER HEAD
# =========================
class CosineClassifier(nn.Module):
    def __init__(self, num_classes, in_features, scale=10.0):
        super().__init__()
        # The weights will hold our prototypes
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        self.scale = scale # Prevents the 64% confidence cap
        
    def forward(self, x):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        # Bounded Cosine Similarity scaled by temperature
        return F.linear(x_norm, w_norm) * self.scale

def test():
    strategy_name = "Personal" if IS_PERSONAL else "General"
    print(f"Starting {N_WAY}-Way {K_SHOT}-Shot {strategy_name} Fine-Tuning Evaluation...")
    
    grand_total_accuracy = 0.0

    # Loop through all 10 seeds
    for current_seed in range(10):
        data_dir = f"dhg_clean/seed_{current_seed}"
        
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

        # Load the base Meta-Trained Model
        base_model = DDNet().to(DEVICE)
        base_model.load_state_dict(torch.load(f"weights/meta_model_seed_{current_seed}.pth"))
        base_model.eval() 
        
        seed_total_acc = 0.0

        for episode, batch in enumerate(loader):
            if episode >= EPISODES:
                break

            # 1. Reset Model for this episode
            model = DDNet().to(DEVICE)
            model.load_state_dict(base_model.state_dict())
            model.train()
            model.apply(set_bn_eval) # Protect BN stats!

            support_x, support_y, query_x, query_y = batch
            support_x = support_x.squeeze(0).to(DEVICE)
            support_y = support_y.squeeze(0).to(DEVICE)
            query_x   = query_x.squeeze(0).to(DEVICE)
            query_y   = query_y.squeeze(0).to(DEVICE)

            # 2. Extract initial prototypes to warm-start the classifier
            with torch.no_grad():
                initial_emb = model(support_x)
                prototypes = compute_prototypes(initial_emb, support_y, N_WAY)
                
            # Initialize the explicit Cosine Classifier head
            classifier = CosineClassifier(N_WAY, 256).to(DEVICE)
            classifier.weight.data = prototypes.data

            # 3. Optimizer controls BOTH the backbone and the new classifier
            optimizer = optim.AdamW(
                list(model.parameters()) + list(classifier.parameters()), 
                lr=5e-3, 
                weight_decay=1e-3
            )

            # 4. Fine-Tune Loop (Stable Gradients)
            for _ in range(FINE_TUNE_STEPS):
                optimizer.zero_grad()
                emb = model(support_x)
                logits = classifier(emb)
                loss = F.cross_entropy(logits, support_y)
                loss.backward()
                optimizer.step()

            # 5. Evaluate on Query Set
            model.eval()
            classifier.eval()
            with torch.no_grad():
                q_emb = model(query_x)
                logits = classifier(q_emb)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == query_y).float().mean()

            seed_total_acc += acc.item()

        # Calculate final accuracy for this single seed
        seed_final_acc = (seed_total_acc / EPISODES) * 100
        print(f"Seed {current_seed} Accuracy: {seed_final_acc:.2f}%")
        
        grand_total_accuracy += seed_final_acc

    # Calculate definitive average
    final_10_seed_average = grand_total_accuracy / 10
    print(f"\n{'='*50}")
    print(f"🏆 FINAL 10-SEED AVERAGE ({strategy_name}): {final_10_seed_average:.2f}%")
    print(f"{'='*50}")

if __name__ == "__main__":
    test()