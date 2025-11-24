import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralDiffusion(nn.Module):
    def __init__(self, feature_dim, num_steps=100):  # Increased steps
        super(SpectralDiffusion, self).__init__()
        
        self.num_steps   = num_steps
        self.feature_dim = feature_dim
        
        # Beta schedule 
        #self.beta = torch.linspace(1e-4, 0.02, num_steps)
        self.beta = torch.logspace(-6, -4, num_steps)

        # Transform network 
        hidden_dim     = 1024  
        self.transform = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Residual projection
        self.residual_proj = nn.Linear(feature_dim * 2, feature_dim)
        
        # FiLM modulation
        self.film_scale = nn.Linear(feature_dim, feature_dim)
        self.film_shift = nn.Linear(feature_dim, feature_dim)

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize network weights 
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def apply_film(self, x, key_vec):
        """
        Apply Feature-wise Linear Modulation based on key vector
        """
        scale = torch.sigmoid(self.film_scale(key_vec)) * 0.5 + 0.5
        shift = self.film_shift(key_vec)
        return x * scale + shift
    
    def unlinkability_loss(self, Z_T1, Z_T2):
        """
        Unlinkability loss for different keys
        """
        sim = F.cosine_similarity(Z_T1.unsqueeze(0), Z_T2.unsqueeze(0), dim=-1)
        return torch.abs(sim)
    
    def key_diversity_loss(self, Z_T1, Z_T2, key1, key2):
        """
        Key diversity loss
        """
        if key1 == key2:
            return torch.tensor(0.0, requires_grad=True, device=Z_T1.device)
        
        sim = F.cosine_similarity(Z_T1.unsqueeze(0), Z_T2.unsqueeze(0), dim=-1)
        return torch.abs(sim)
    
    def diversity_loss(self, Z_T_batch):
        """
        Promote diversity in protected embeddings
        """
        if Z_T_batch.size(0) < 2:
            return torch.tensor(0.0, requires_grad=True, device=Z_T_batch.device)
        
        sim_matrix = F.cosine_similarity(Z_T_batch.unsqueeze(1), Z_T_batch.unsqueeze(0), dim=-1)
        return -torch.mean(sim_matrix.triu(diagonal=1))
    
    def discriminability_loss(self, Z_batch, Z_T_batch, labels_Z, labels_Z_T, key_batch):
        """
        Discriminability loss to preserve similarity structure
        """
        batch_size = Z_batch.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, requires_grad=True, device=Z_batch.device)
        
        # Calculate similarity matrices
        orig_sim = F.cosine_similarity(Z_batch.unsqueeze(1),     Z_batch.unsqueeze(0), dim=-1)
        prot_sim = F.cosine_similarity(Z_T_batch.unsqueeze(1), Z_T_batch.unsqueeze(0), dim=-1)
        
        # Prepare labels and keys
        labels_Z   = labels_Z.view(-1, 1)
        labels_Z_T = labels_Z_T.view(-1, 1)
        key_batch  = key_batch.view(-1, 1) if key_batch.dim() == 1 else key_batch
        
        # Create masks
        same_label_orig = (labels_Z == labels_Z.t()).float().triu(diagonal=1)
        same_label_prot = (labels_Z_T == labels_Z_T.t()).float().triu(diagonal=1)
        diff_label_prot = (labels_Z_T != labels_Z_T.t()).float().triu(diagonal=1)
        same_key = (key_batch == key_batch.t()).float().triu(diagonal=1)
        diff_key = (key_batch != key_batch.t()).float().triu(diagonal=1)
        
        # Scenario masks
        match_same_key_mask    = same_label_prot * same_key
        mismatch_same_key_mask = diff_label_prot * same_key
        match_diff_key_mask    = same_label_prot * diff_key
        
        # Preserve similarity for genuine pairs
        pres_mask = same_label_orig * match_same_key_mask
        dist_preservation = torch.sum(pres_mask * (orig_sim - prot_sim)**2) / (torch.sum(pres_mask) + 1e-8)
        
        # Push different-label, same-key pairs apart
        mismatch_same_key_term = torch.sum(mismatch_same_key_mask * prot_sim**2) / (torch.sum(mismatch_same_key_mask) + 1e-8)
        
        # Push same-label, different-key pairs apart
        match_diff_key_term = torch.sum(match_diff_key_mask * prot_sim**2) / (torch.sum(match_diff_key_mask) + 1e-8)
        
        # Reduced weights to balance discriminability
        #total_loss = dist_preservation + 2.0 * mismatch_same_key_term + 1.0 * match_diff_key_term
        total_loss = dist_preservation + 7.0 * mismatch_same_key_term + 1.0 * match_diff_key_term

        return total_loss
    
    def contrastive_loss(self, Z_T_batch, labels_Z_T, key_batch, margin=0.2):  # Increased margin
        """
        Contrastive loss for pair-wise relationships
        """
        batch_size = Z_T_batch.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, requires_grad=True, device=Z_T_batch.device)
        
        prot_sim   = F.cosine_similarity(Z_T_batch.unsqueeze(1), Z_T_batch.unsqueeze(0), dim=-1)
        
        labels_Z_T = labels_Z_T.view(-1, 1)
        key_batch  = key_batch.view(-1, 1) if key_batch.dim() == 1 else key_batch
        
        same_label = (labels_Z_T == labels_Z_T.t()).float().triu(diagonal=1)
        same_key   = (key_batch == key_batch.t()).float().triu(diagonal=1)
        
        match_same_key_mask    = same_label * same_key
        mismatch_same_key_mask = (1 - same_label) * same_key
        match_diff_key_mask    = same_label * (1 - same_key)
        mismatch_diff_key_mask = (1 - same_label) * (1 - same_key)
        
        match_same_key_loss    = torch.sum(match_same_key_mask * (1.0 - prot_sim)) / (torch.sum(match_same_key_mask) + 1e-8)
        mismatch_same_key_loss = torch.sum(mismatch_same_key_mask * F.relu(prot_sim - margin)) / (torch.sum(mismatch_same_key_mask) + 1e-8)
        match_diff_key_loss    = torch.sum(match_diff_key_mask * F.relu(prot_sim - margin)) / (torch.sum(match_diff_key_mask) + 1e-8)
        mismatch_diff_key_loss = torch.sum(mismatch_diff_key_mask * F.relu(prot_sim - margin)) / (torch.sum(mismatch_diff_key_mask) + 1e-8)
        
        # Adjusted weights
        #total_loss = 3.0 * match_same_key_loss + 2.0 * mismatch_same_key_loss + 1.0 * match_diff_key_loss + 1.0 * mismatch_diff_key_loss
        total_loss = 1.0 * match_same_key_loss + 6.0 * mismatch_same_key_loss + 1.0 * match_diff_key_loss + 1.0 * mismatch_diff_key_loss
        
        return total_loss
    
    def generate_key_vector(self, seed, feature_dim):
        """
        Generate deterministic key vector from seed
        """
        torch.manual_seed(seed)
        key_vec = torch.randn(feature_dim)
        return F.normalize(key_vec, p=2, dim=-1)
    
    def forward(self, Z, key):
        """
        Forward diffusion process
        """
        Z_t = Z.clone()

        if Z_t.dim() == 2:
            Z_t = Z_t.squeeze(0)
     
        key_vec = self.generate_key_vector(key, self.feature_dim).to(Z.device)
        # Store original norms
        norms   = torch.norm(Z, p=2, dim=-1, keepdim=True)
        
        for t in range(self.num_steps):
            noise           = torch.randn_like(Z_t) * torch.sqrt(self.beta[t].to(Z.device))
            transform_input = torch.cat([Z_t, key_vec], dim=-1)

            x          = self.transform(transform_input)
            phi_output = x + self.residual_proj(transform_input)
            phi_output = self.apply_film(phi_output, key_vec)
            
            Z_t = Z_t + (1 - self.beta[t].to(Z.device)) * phi_output + noise  
            
            # Selective normalization (only every 5 steps)
            if (t + 1) % 5 == 0:
                Z_t = F.normalize(Z_t, p=2, dim=-1) * norms
                if Z_t.dim() == 2:
                    Z_t = Z_t.squeeze(0)
         
        return F.normalize(Z_t, p=2, dim=-1)

    def reverse(self, Z_T, key):
        """
        Reverse diffusion (for evaluation, not training)
        """
        Z_t = Z_T.clone()

        if Z_t.dim() == 1:
            Z_t = Z_t.squeeze(0)
        
        key_vec = self.generate_key_vector(key, self.feature_dim).to(Z_T.device)
        norms   = torch.norm(Z_T, p=2, dim=-1, keepdim=True)
        
        for t in reversed(range(self.num_steps)):
            noise           = torch.randn_like(Z_t) * torch.sqrt(self.beta[t].to(Z_T.device))
            transform_input = torch.cat([Z_t, key_vec], dim=-1)

            x          = self.transform(transform_input)
            phi_output = x + self.residual_proj(transform_input)
            phi_output = self.apply_film(phi_output, key_vec)
            
            Z_t = Z_t - (1 - self.beta[t].to(Z_T.device)) * phi_output - noise
            
            if (t + 1) % 5 == 0:
                Z_t = F.normalize(Z_t, p=2, dim=-1) * norms
                if Z_t.dim() == 2:
                    Z_t = Z_t.squeeze(0)
        
        return F.normalize(Z_t, p=2, dim=-1)





























































"""
class SpectralDiffusion(nn.Module):
    def __init__(self, feature_dim, num_steps=20):
        super(SpectralDiffusion, self).__init__()
        
        self.num_steps   = num_steps
        self.feature_dim = feature_dim
        
        # Values between 0.0001 and 0.02 work well for diffusion models
        #self.beta = torch.linspace(0.0001, 0.02, num_steps)
        self.beta = torch.logspace(-6, -4, num_steps)
        #self.beta = torch.logspace(-10, -5, num_steps)

        # Transform network
        hidden_dim = 1024
        self.transform = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Residual projection
        self.residual_proj = nn.Linear(feature_dim * 2, feature_dim)
        
        # Key embedding
        self.key_embed = nn.Linear(1, feature_dim)
        
        # FiLM modulation
        self.film_scale = nn.Linear(feature_dim, feature_dim)
        self.film_shift = nn.Linear(feature_dim, feature_dim)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def apply_film(self, x, key_vec):
        scale = torch.sigmoid(self.film_scale(key_vec)) * 0.5 + 0.5
        shift = self.film_shift(key_vec)
        return x * scale + shift
    
    def diversity_loss(self, Z_T_batch):
        if Z_T_batch.size(0) < 2:
            return torch.tensor(0.0, requires_grad=True, device=Z_T_batch.device)
        sim_matrix = F.cosine_similarity(Z_T_batch.unsqueeze(1), Z_T_batch.unsqueeze(0), dim=-1)
        return -torch.mean(sim_matrix.triu(diagonal=1))
    
    def unlinkability_loss(self, Z_T1, Z_T2):
        sim = F.cosine_similarity(Z_T1.unsqueeze(0), Z_T2.unsqueeze(0), dim=-1)
        return 2.0 * torch.abs(sim)
    
    def key_diversity_loss(self, Z_T1, Z_T2, key1, key2):
        if key1 == key2:
            return torch.tensor(0.0, requires_grad=True, device=Z_T1.device)
        sim = F.cosine_similarity(Z_T1.unsqueeze(0), Z_T2.unsqueeze(0), dim=-1)
        return torch.abs(sim)
    
    def discriminability_loss(self, Z_batch, Z_T_batch, labels):
        batch_size = Z_batch.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, requires_grad=True, device=Z_batch.device)
        
        if Z_T_batch.size(0) != batch_size or labels.size(0) != batch_size:
            raise ValueError(
                f"Inconsistent batch sizes: Z_batch={batch_size}, "
                f"Z_T_batch={Z_T_batch.size(0)}, labels={labels.size(0)}"
            )
        
        orig_sim = F.cosine_similarity(
            Z_batch.unsqueeze(1), Z_batch.unsqueeze(0), dim=-1
        )
        prot_sim = F.cosine_similarity(
            Z_T_batch.unsqueeze(1), Z_T_batch.unsqueeze(0), dim=-1
        )
        prot_sim = prot_sim.squeeze(-1)

        if orig_sim.shape != (batch_size, batch_size) or prot_sim.shape != (batch_size, batch_size):
            raise ValueError(
                f"Expected similarity matrices of shape [{batch_size}, {batch_size}], "
                f"got orig_sim={orig_sim.shape}, prot_sim={prot_sim.shape}"
            )
        
        diff = (orig_sim.triu(diagonal=1) - prot_sim.triu(diagonal=1)) ** 2
        
        labels = labels.view(-1, 1)
        match_mask = (labels == labels.t()).float().triu(diagonal=1)
        mismatch_mask = (1 - match_mask).triu(diagonal=1)
        
        mismatch_sim_loss = torch.mean(mismatch_mask * torch.clamp(prot_sim, min=0))
        
        return torch.mean(diff) + 2.0 * mismatch_sim_loss
    
    def contrastive_loss(self, Z_T_batch, labels, margin=0.1):
        batch_size = Z_T_batch.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, requires_grad=True, device=Z_T_batch.device)
        
        if labels.size(0) != batch_size:
            raise ValueError(
                f"Inconsistent batch sizes: Z_T_batch={batch_size}, labels={labels.size(0)}"
            )
        
        sim_matrix = F.cosine_similarity(
            Z_T_batch.unsqueeze(1), Z_T_batch.unsqueeze(0), dim=-1
        )
        sim_matrix = sim_matrix.squeeze(-1)           

        if sim_matrix.shape != (batch_size, batch_size):
            raise ValueError(
                f"Expected similarity matrix of shape [{batch_size}, {batch_size}], "
                f"got {sim_matrix.shape}"
            )
        
        labels = labels.view(-1, 1)
        match_mask = (labels == labels.t()).float().triu(diagonal=1)
        mismatch_mask = (1 - match_mask).triu(diagonal=1)
        
        match_loss = torch.mean(match_mask * (1 - sim_matrix))
        mismatch_loss = torch.mean(mismatch_mask * F.relu(sim_matrix - margin))
        
        return match_loss + 2.0 * mismatch_loss
    
    def distortion_loss(self, Z_T):
        return torch.mean(torch.abs(Z_T - Z_T.mean(dim=0)))
    
    def generate_key_vector(self, seed, feature_dim):
        torch.manual_seed(seed)
        key_vec = torch.randn(feature_dim)
        return F.normalize(key_vec, p=2, dim=-1)
    
    def forward(self, Z, key):
        # Ensure Z is 2D
        if Z.dim() == 1:
            Z = Z.unsqueeze(0)
        elif Z.dim() != 2:
            raise ValueError(f"Expected Z to have 1 or 2 dimensions, got {Z.dim()}")
        
        batch_size = Z.size(0)
        if Z.size(1) != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {Z.size(1)}")
        
        Z = F.normalize(Z, p=2, dim=-1)
        Z_t = Z.clone()
        key_vec = self.generate_key_vector(key, self.feature_dim).to(Z.device)

        key_vec = key_vec.unsqueeze(0)
        
        if key_vec.size(0) != batch_size or key_vec.size(1) != self.feature_dim:
            raise ValueError(
                f"Expected key_vec shape [{batch_size}, {self.feature_dim}], got {key_vec.shape}"
            )
        
        for t in range(self.num_steps):
            torch.manual_seed(int(key[0]) + t if isinstance(key, (list, tuple, torch.Tensor)) else key + t)
            noise = torch.randn_like(Z_t) * torch.sqrt(self.beta[t].to(Z.device))
            transform_input = torch.cat([Z_t, key_vec], dim=-1)
            
            x = self.transform(transform_input)
            phi_output = x + self.residual_proj(transform_input)
            phi_output = self.apply_film(phi_output, key_vec)
            
            Z_t = Z_t + noise + (1 - self.beta[t].to(Z.device)) * phi_output
            
            # Ensure Z_t shape before normalization
            if Z_t.shape != (batch_size, self.feature_dim):
                raise ValueError(
                    f"Expected Z_t shape [{batch_size}, {self.feature_dim}], got {Z_t.shape}"
                )
            
            # Normalize and scale
            norms = torch.norm(Z, p=2, dim=-1).view(-1, 1)  # Shape: [batch_size, 1]
            Z_t = F.normalize(Z_t, p=2, dim=-1) * norms
            
            if Z_t.shape != (batch_size, self.feature_dim):
                raise ValueError(
                    f"Post-normalization Z_t shape [{batch_size}, {self.feature_dim}], got {Z_t.shape}"
                )
        Z_t = Z_t.squeeze(0)
        return F.normalize(Z_t, p=2, dim=-1)

    def reverse(self, Z_T, key):
        if Z_T.dim() == 1:
            Z_T = Z_T.unsqueeze(0)
        elif Z_T.dim() != 2:
            raise ValueError(f"Expected Z_T to have 1 or 2 dimensions, got {Z_T.dim()}")
        
        batch_size = Z_T.size(0)
        if Z_T.size(1) != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {Z_T.size(1)}")
        
        Z_T = F.normalize(Z_T, p=2, dim=-1)
        Z_t = Z_T.clone()
        key_vec = self.generate_key_vector(key, self.feature_dim).to(Z_T.device)
        key_vec = key_vec.unsqueeze(0)

        if key_vec.size(0) != batch_size or key_vec.size(1) != self.feature_dim:
            raise ValueError(
                f"Expected key_vec shape [{batch_size}, {self.feature_dim}], got {key_vec.shape}"
            )
        
        for t in reversed(range(self.num_steps)):
            torch.manual_seed(int(key[0]) + t if isinstance(key, (list, tuple, torch.Tensor)) else key + t)
            noise = torch.randn_like(Z_t) * torch.sqrt(self.beta[t].to(Z_T.device))
            transform_input = torch.cat([Z_t, key_vec], dim=-1)
            
            x = self.transform(transform_input)
            phi_output = x + self.residual_proj(transform_input)
            phi_output = self.apply_film(phi_output, key_vec)
            
            Z_t = Z_t - noise - (1 - self.beta[t].to(Z_T.device)) * phi_output
            
            if Z_t.shape != (batch_size, self.feature_dim):
                raise ValueError(
                    f"Expected Z_t shape [{batch_size}, {self.feature_dim}], got {Z_t.shape}"
                )
            
            norms = torch.norm(Z_T, p=2, dim=-1).view(-1, 1)
            Z_t = F.normalize(Z_t, p=2, dim=-1) * norms
            
            if Z_t.shape != (batch_size, self.feature_dim):
                raise ValueError(
                    f"Post-normalization Z_t shape [{batch_size}, {self.feature_dim}], got {Z_t.shape}"
                )
        Z_t = Z_t.squeeze(0)           
        return F.normalize(Z_t, p=2, dim=-1)

"""










