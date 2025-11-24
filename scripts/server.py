import torch
import torch.nn.functional as F
import numpy as np

# Server-Side Implementation
class Server:
    def __init__(self):
        self.storage = {}

    def enroll(self, user_id, Z_T):
        if user_id in self.storage:
            raise ValueError(f"User {user_id} already enrolled")
        
        self.storage[user_id] = Z_T
        print(f"Enrolled user {user_id}.")

    def match(self, Z_query, user_id, threshold=0.7):
        if user_id not in self.storage:
            raise ValueError(f"User {user_id} not enrolled.")
        
        Z_T_enrolled = self.storage[user_id]
        similarity   = F.cosine_similarity(Z_query.unsqueeze(0), Z_T_enrolled.unsqueeze(0)).item()
        return similarity > threshold, similarity
        
        