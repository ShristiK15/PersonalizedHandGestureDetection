import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# =========================
# PATH SETUP
# =========================
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from meta_ddnet_model import DDNet
from meta_dataset import FewShotDataset
from proto_net import compute_prototypes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CONFIG
# =========================
N_WAY = 5
K_SHOT = 1
IS_PERSONAL = False


if IS_PERSONAL:
    K_QUERY = 5 - K_SHOT
else:
    K_QUERY = 15

FINE_TUNE_STEPS = 10
EPISODES = 150

# =========================
# BATCHNORM FREEZE
# =========================
def set_bn_eval(m):
    if isinstance(m, nn.BatchNorm1d):
        m.eval()

# =========================
# COSINE CLASSIFIER 
# =========================
class CosineClassifier(nn.Module):
    def __init__(self, num_classes, in_features, scale=5.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        self.scale = scale

    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return F.linear(x, w) * self.scale

# =========================
# TEST FUNCTION
# =========================
def test():
    print(f"\n🚀 Starting {N_WAY}-Way {K_SHOT}-Shot Evaluation...\n")

    grand_total_acc = 0.0

    for seed in range(10):
        print(f"\n===== SEED {seed} =====")

        data_dir = f"dhg_clean/seed_{seed}"

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

        # Load meta-trained model
        base_model = DDNet().to(DEVICE)
        base_model.load_state_dict(torch.load(f"meta_weights/meta_model_seed_{seed}.pth"))
        base_model.eval()

        total_acc = 0.0

        for episode, batch in enumerate(loader):
            if episode >= EPISODES:
                break

            # =========================
            # RESET MODEL
            # =========================
            model = DDNet().to(DEVICE)
            model.load_state_dict(base_model.state_dict())

            model.train()
            model.apply(set_bn_eval)

            support_x, support_y, query_x, query_y = batch

            support_x = support_x.squeeze(0).to(DEVICE)
            support_y = support_y.squeeze(0).to(DEVICE)
            query_x   = query_x.squeeze(0).to(DEVICE)
            query_y   = query_y.squeeze(0).to(DEVICE)

            # =========================
            # INIT PROTOTYPES
            # =========================
            with torch.no_grad():
                emb = model(support_x)
                emb = F.normalize(emb, dim=1)
                prototypes = compute_prototypes(emb, support_y, N_WAY)

            classifier = CosineClassifier(N_WAY, 256).to(DEVICE)
            classifier.weight.data = prototypes.data

            # =========================
            # OPTIMIZER (IMPORTANT)
            # =========================
            optimizer = optim.AdamW([
                {"params": model.parameters(), "lr": 5e-4},
                {"params": classifier.parameters(), "lr": 1e-3}
            ], weight_decay=1e-3)

            # =========================
            # FINETUNING
            # =========================
            model.train()
            classifier.train()

            for _ in range(FINE_TUNE_STEPS):
                optimizer.zero_grad()

                emb = model(support_x)
                emb = F.normalize(emb, dim=1)

                logits = classifier(emb)
                loss = F.cross_entropy(logits, support_y)

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

            # =========================
            # TESTING
            # =========================
            model.eval()
            classifier.eval()

            with torch.no_grad():
                q_emb = model(query_x)
                q_emb = F.normalize(q_emb, dim=1)

                logits = classifier(q_emb)
                preds = torch.argmax(logits, dim=1)

                acc = (preds == query_y).float().mean()

            total_acc += acc.item()

        seed_acc = (total_acc / EPISODES) * 100
        print(f"Seed {seed} Accuracy: {seed_acc:.2f}%")

        grand_total_acc += seed_acc

    final_acc = grand_total_acc / 10

    print("\n" + "="*50)
    print(f"🏆 FINAL ACCURACY: {final_acc:.2f}%")
    print("="*50)


if __name__ == "__main__":
    test()