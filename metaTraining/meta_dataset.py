# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import random

# class FewShotDataset(Dataset):
#     def __init__(self, data_path, label_path, n_way=5, k_shot=1, k_query=15):
#         self.data = np.load(data_path)
#         self.labels = np.load(label_path)

#         self.n_way = n_way
#         self.k_shot = k_shot
#         self.k_query = k_query

#         self.classes = list(set(self.labels))
#         self.class_to_indices = {
#             c: np.where(self.labels == c)[0] for c in self.classes
#         }

#     def __len__(self):
#         return 1000  # number of episodes

#     def __getitem__(self, idx):
#         selected_classes = random.sample(self.classes, self.n_way)

#         support_x, support_y = [], []
#         query_x, query_y = [], []

#         for i, cls in enumerate(selected_classes):
#             indices = self.class_to_indices[cls]

#             selected = np.random.choice(
#                 indices,
#                 self.k_shot + self.k_query,
#                 replace=False
#             )

#             support_idx = selected[:self.k_shot]
#             query_idx   = selected[self.k_shot:]

#             for s in support_idx:
#                 support_x.append(self.data[s])
#                 support_y.append(i)

#             for q in query_idx:
#                 query_x.append(self.data[q])
#                 query_y.append(i)

#         return (
#             torch.tensor(support_x, dtype=torch.float32),
#             torch.tensor(support_y),
#             torch.tensor(query_x, dtype=torch.float32),
#             torch.tensor(query_y)
#         )


import numpy as np
import torch
from torch.utils.data import Dataset
import random

class FewShotDataset(Dataset):
    def __init__(self, data_path, label_path, user_path=None, n_way=5, k_shot=1, k_query=15, is_personal=False):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        self.is_personal = is_personal

        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query

        self.classes = list(set(self.labels))

        if self.is_personal:
            # Personal Strategy: We need user data to lock episodes to specific users
            assert user_path is not None, "user_path must be provided when is_personal=True"
            self.users = np.load(user_path)
            self.unique_users = list(set(self.users))
            
            # Create a lookup dictionary mapping (user, class) -> list of indices
            self.user_class_to_indices = {}
            for u in self.unique_users:
                for c in self.classes:
                    # Find indices where both the user and the class match
                    idx = np.where((self.users == u) & (self.labels == c))[0]
                    if len(idx) > 0:
                        self.user_class_to_indices[(u, c)] = idx
        else:
            # General Strategy: We only care about classes
            self.class_to_indices = {
                c: np.where(self.labels == c)[0] for c in self.classes
            }

    def __len__(self):
        return 1000  # Number of episodes

    def __getitem__(self, idx):
        selected_classes = random.sample(self.classes, self.n_way)

        support_x, support_y = [], []
        query_x, query_y = [], []

        # If personal, select ONE user for this entire episode
        if self.is_personal:
            selected_user = random.choice(self.unique_users)

        for i, cls in enumerate(selected_classes):
            
            # Fetch the valid indices based on the strategy
            if self.is_personal:
                indices = self.user_class_to_indices.get((selected_user, cls), [])
                if len(indices) < (self.k_shot + self.k_query):
                    raise ValueError(f"User {selected_user} doesn't have enough samples for class {cls}. Need {self.k_shot + self.k_query}, got {len(indices)}.")
            else:
                indices = self.class_to_indices[cls]

            # Randomly pick the exact number of samples needed for support + query
            selected = np.random.choice(
                indices,
                self.k_shot + self.k_query,
                replace=False
            )

            support_idx = selected[:self.k_shot]
            query_idx   = selected[self.k_shot:]

            for s in support_idx:
                support_x.append(self.data[s])
                support_y.append(i)

            for q in query_idx:
                query_x.append(self.data[q])
                query_y.append(i)

        # Convert to arrays first for faster PyTorch tensor creation
        return (
            torch.tensor(np.array(support_x), dtype=torch.float32),
            torch.tensor(support_y),
            torch.tensor(np.array(query_x), dtype=torch.float32),
            torch.tensor(query_y)
        )