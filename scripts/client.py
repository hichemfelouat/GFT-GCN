import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import entropy, pearsonr, ttest_1samp
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

import time
from scripts.gcn import*
from scripts.diffusion import*
from scripts.entropy import*

from scripts.preimageattack import*

# Client-Side Implementation 
class Client(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=16, diffusion_steps=50, threshold=0.9):
        super(Client, self).__init__()
        self.threshold = threshold

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gcn       = GCN(in_channels, hidden_channels, out_channels).to(self.device)
        self.diffusion = SpectralDiffusion(out_channels, diffusion_steps).to(self.device)

        self.gcn_optimizer       = torch.optim.Adam(self.gcn.parameters(), lr=0.001)
        #self.diffusion_optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=0.0001)
        self.diffusion_optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=0.0001, weight_decay=1e-4)
  
    def extract_features(self, x, edge_index):
        """
        Extract features using a GCN on spectral features.
        
        Parameters:
        -----------
        x : torch.Tensor
            Spectral features tensor with shape [k, feature_dim]
        edge_index : torch.Tensor
            Edge index tensor for the original graph
            
        Returns:
        --------
        torch.Tensor
            Normalized feature vector
        """
        # Get the number of spectral components
        k = x.shape[0]
        
        # Create a line graph connecting consecutive spectral components
        spectral_edge_index = torch.tensor([[i, i+1] for i in range(k-1)] +
                                          [[i+1, i] for i in range(k-1)],
                                          dtype=torch.long).t().contiguous()
        
        # Ensure x has the same dtype as expected by the GCN (typically float32)
        x = x.to(torch.float32)
        
        # Make sure edge_index is on the same device as x
        spectral_edge_index = spectral_edge_index.to(x.device)
        
        # Apply GCN and average features across nodes
        features = self.gcn(x, spectral_edge_index).mean(dim=0)
        
        # Return L2 normalized features
        return F.normalize(features, p=2, dim=-1)

    def protect_template(self, Z, key):
        return self.diffusion(Z, key)

    def reconstruct_template(self, Z_T, key):
        return self.diffusion.reverse(Z_T, key)

    def process_mesh(self, x, edge_index, key, is_enrollment=False):
        if x is None or edge_index is None:
            raise ValueError("Input mesh data (x, edge_index) cannot be None")
        Z   = self.extract_features(x, edge_index)
        Z_T = self.protect_template(Z, key)
        return Z_T if is_enrollment else self.reconstruct_template(Z_T, key)
      
    def train_gcn(self, train_data, val_data, num_epochs=50, margin=0.9, batch_size=16):
        """
        Train only the GCN for binary classification.
        """
        print("Stage 1: Training GCN for binary classification...")
        losses       = {'siamese': []}
        best_val_acc = 0.0
        best_model   = None
        self.gcn.train()

        for epoch in range(num_epochs):
            epoch_siamese_loss = 0.0
            batch_data         = []

            for i, (x1, edge_index1, x2, edge_index2, label) in enumerate(train_data):
                batch_data.append((x1, edge_index1, x2, edge_index2, label))

                if len(batch_data) == batch_size or i == len(train_data) - 1:
                    self.gcn_optimizer.zero_grad()
                    batch_siamese_loss = 0.0
                    
                    # Process batch
                    for x1_b, edge_index1_b, x2_b, edge_index2_b, label_b in batch_data:
                        Z1 = self.extract_features(x1_b, edge_index1_b)
                        Z2 = self.extract_features(x2_b, edge_index2_b)

                        # Siamese loss
                        s_loss = self.gcn.siamese_loss(Z1, Z2, label_b, margin)
                        batch_siamese_loss += s_loss

                    # Backpropagate once per batch
                    batch_siamese_loss /= len(batch_data)
                    batch_siamese_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.gcn.parameters(), max_norm=1.0)
                    self.gcn_optimizer.step()

                    epoch_siamese_loss += batch_siamese_loss.item() * len(batch_data)
                    batch_data          = []  # Reset batch

            # Average losses over epoch
            n_samples = len(train_data)
            losses['siamese'].append(epoch_siamese_loss / n_samples)

            val_acc = self.validate_gcn(val_data)
            print(f"Epoch {epoch + 1}/{num_epochs}, Siamese Loss: {losses['siamese'][-1]:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model   = self.gcn.state_dict().copy()

        # Load best model
        if best_model:
            self.gcn.load_state_dict(best_model)
            print(f"Loaded best GCN model with validation accuracy: {best_val_acc:.4f}")

        # Freeze GCN parameters
        for param in self.gcn.parameters():
            param.requires_grad = False

        return losses

    def validate_gcn(self, val_data):
        """
        Validate GCN performance on binary classification.
        """
        self.gcn.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for x1, edge_index1, x2, edge_index2, label in val_data:
                Z1 = self.extract_features(x1, edge_index1)
                Z2 = self.extract_features(x2, edge_index2)

                # Calculate cosine similarity
                sim  = F.cosine_similarity(Z1.unsqueeze(0), Z2.unsqueeze(0)).item()
                pred = 1 if sim > self.threshold else 0

                if pred == label:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0
    
    def get_key(self):
        return torch.randint(0, 1000000, (1,)).item()
    
    #****************************************************************************
    #****************************************************************************
    def train_diffusion(self, train_data, val_data, num_epochs=100, batch_size=32):  
        # Train the Spectral Diffusion model with frozen GCN
        print("Stage 2: Training Spectral Diffusion with frozen GCN...")
        
        losses = {
            'recon'           : [],
            'diversity'       : [],
            'discriminability': [],
            'contrastive'     : [],
            'unlinkability'   : [],
            'key_diversity'   : [],
            'total'           : []
        }

        best_val_score = float('inf')
        best_model     = None
        best_epoch     = -1
        
        self.gcn.eval()
        self.diffusion.train()
        
        for epoch in range(num_epochs):
            epoch_div_loss = epoch_disc_loss = epoch_cont_loss = epoch_unlink_loss = epoch_key_div_loss = epoch_total_loss = 0.0
            epoch_recon_loss = 0.0
            batch_data = []
            
            for i, (x1, edge_index1, x2, edge_index2, label) in enumerate(train_data):
                batch_data.append((x1, edge_index1, x2, edge_index2, label))
                
                if len(batch_data) == batch_size or i == len(train_data) - 1:
                    self.diffusion_optimizer.zero_grad()
                    
                    Z_batch_same    = []
                    Z_T_batch_same  = []
                    labels_Z_same   = []
                    labels_Z_T_same = []
                    key_batch_same  = []
                    
                    Z_T_batch_diff  = []
                    labels_Z_T_diff = []
                    key_batch_diff  = []
                    
                    batch_total_loss = 0.0
                    
                    for x1_b, edge_index1_b, x2_b, edge_index2_b, label_b in batch_data:
                        with torch.no_grad():
                            Z1 = self.extract_features(x1_b, edge_index1_b)
                            Z2 = self.extract_features(x2_b, edge_index2_b)
                        
                        key  = self.get_key()
                        Z_T1 = self.protect_template(Z1, key)
                        Z_T2 = self.protect_template(Z2, key)
                        
                        key_diff = self.get_key()
                        while key_diff == key:
                            key_diff = self.get_key()
                        
                        Z_T1_diff = self.protect_template(Z1, key_diff)
                        
                        # Same-key batch
                        Z_batch_same.append(Z1)
                        Z_batch_same.append(Z2)

                        labels_Z_same.append(label_b)
                        labels_Z_same.append(label_b)

                        Z_T_batch_same.append(Z_T1)
                        Z_T_batch_same.append(Z_T2)
                        labels_Z_T_same.append(label_b)
                        labels_Z_T_same.append(label_b)

                        key_batch_same.append(key)
                        key_batch_same.append(key)
                        
                        # Different-key batch
                        Z_T_batch_diff.append(Z_T1)
                        Z_T_batch_diff.append(Z_T1_diff)

                        labels_Z_T_diff.append(label_b)
                        labels_Z_T_diff.append(label_b)

                        key_batch_diff.append(key)
                        key_batch_diff.append(key_diff)
                        
                        #*******************************************************************
                        Z_recon1 = self.reconstruct_template(Z_T1, key)
                        Z_recon2 = self.reconstruct_template(Z_T2, key)
                        r_loss   = (F.mse_loss(Z_recon1, Z1) + F.mse_loss(Z_recon2, Z2)) / 2
 
                        unlink_loss  = self.diffusion.unlinkability_loss(Z_T1, Z_T1_diff)
                        key_div_loss = self.diffusion.key_diversity_loss(Z_T1, Z_T1_diff, key, key_diff)
                        
                        batch_loss = 0.5 * unlink_loss + 0.3 * key_div_loss + r_loss 
                        #*******************************************************************
                        epoch_recon_loss   += r_loss.item()
                        epoch_unlink_loss  += unlink_loss.item()  if label_b == 1 else 0.0
                        epoch_key_div_loss += key_div_loss.item() if label_b == 1 else 0.0
                    
                    if len(Z_T_batch_same) >= 2:
                        Z_batch_same    = torch.stack(Z_batch_same).to(self.device)
                        Z_T_batch_same  = torch.stack(Z_T_batch_same).to(self.device)
                        labels_Z_same   = torch.tensor(labels_Z_same).to(self.device)
                        labels_Z_T_same = torch.tensor(labels_Z_T_same).to(self.device)
                        key_batch_same  = torch.tensor(key_batch_same).to(self.device)
                        
                        Z_T_batch_diff  = torch.stack(Z_T_batch_diff).to(self.device)
                        labels_Z_T_diff = torch.tensor(labels_Z_T_diff).to(self.device)
                        key_batch_diff  = torch.tensor(key_batch_diff).to(self.device)
                        
                        #*******************************************************************
                        disc_loss_same_key = self.diffusion.discriminability_loss(
                                Z_batch_same, Z_T_batch_same, labels_Z_same, labels_Z_T_same, key_batch_same)
                        disc_loss_diff_key = self.diffusion.discriminability_loss(
                                Z_batch_same, Z_T_batch_diff, labels_Z_same, labels_Z_T_diff, key_batch_diff)
                        disc_loss = 3.5 * disc_loss_same_key + 0.5 * disc_loss_diff_key  # Adjusted weights
                        
                        cont_loss_same_key = self.diffusion.contrastive_loss(Z_T_batch_same, labels_Z_T_same, key_batch_same, 0.1)
                        cont_loss_diff_key = self.diffusion.contrastive_loss(Z_T_batch_diff, labels_Z_T_diff, key_batch_diff, 0.1)
                        cont_loss          = 3.5 * cont_loss_same_key + 0.5 * cont_loss_diff_key  # Adjusted weights
                        
                        div_loss1 = self.diffusion.diversity_loss(Z_T_batch_same)
                        div_loss2 = self.diffusion.diversity_loss(Z_T_batch_diff)
                        div_loss  = 0.01 * (div_loss1 + div_loss2)  
                        
                        batch_loss += 1.0 * disc_loss + 1.0 * cont_loss + div_loss
                        #*******************************************************************

                        epoch_div_loss  += div_loss.item()
                        epoch_disc_loss += disc_loss.item()
                        epoch_cont_loss += cont_loss.item()
                    
                    batch_total_loss += batch_loss
                    batch_total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=1.0)
                    self.diffusion_optimizer.step()
                    
                    epoch_total_loss += batch_total_loss.item() * len(batch_data)
                    
                    batch_data = []
            
            n_samples = len(train_data)
            losses['recon'].append(epoch_recon_loss / n_samples)
            losses['diversity'].append(epoch_div_loss / n_samples         if epoch_div_loss > 0     else 0)
            losses['discriminability'].append(epoch_disc_loss / n_samples if epoch_disc_loss > 0    else 0)
            losses['contrastive'].append(epoch_cont_loss / n_samples      if epoch_cont_loss > 0    else 0)
            losses['unlinkability'].append(epoch_unlink_loss / n_samples  if epoch_unlink_loss > 0  else 0)
            losses['key_diversity'].append(epoch_key_div_loss / n_samples if epoch_key_div_loss > 0 else 0)
            losses['total'].append(epoch_total_loss / n_samples)
            
            #*******************************************************************
            val_metrics            = self.validate_diffusion(val_data)

            match_mean_same_key    = val_metrics['match_mean_same_key']
            mismatch_mean_same_key = val_metrics['mismatch_mean_same_key']
            match_mean_diff_key    = val_metrics['match_mean_diff_key']
            mismatch_mean_diff_key = val_metrics['mismatch_mean_diff_key']
            
            #val_score = abs(match_mean_same_key - 1) + 2.0 * abs(mismatch_mean_same_key) + 2.0 * abs(match_mean_diff_key) + abs(mismatch_mean_diff_key)
            val_score = abs(match_mean_same_key - 1) + 2.5 * abs(mismatch_mean_same_key) + 2.0 * abs(match_mean_diff_key) + abs(mismatch_mean_diff_key)
            #*******************************************************************

            print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {losses['total'][-1]:.4f}, "
                f"Match Mean Same Key: {match_mean_same_key:.4f}, Mismatch Mean Same Key: {mismatch_mean_same_key:.4f}, "
                f"Match Mean Diff Key: {match_mean_diff_key:.4f}, Mismatch Mean Diff Key: {mismatch_mean_diff_key:.4f}, "
                f"Val Score: {val_score:.4f}")
            
            if val_score <= best_val_score:
                best_val_score = val_score
                best_model     = self.diffusion.state_dict().copy()
                best_epoch     = epoch
        
        if best_model:
            self.diffusion.load_state_dict(best_model)
            print("-" * 20)
            best_epoch += 1
            print(f"Loaded best diffusion model with validation score: {best_val_score:.4f}, epoch: {best_epoch}")
        
        return losses
    
    #****************************************************************************
    #****************************************************************************
    def validate_diffusion(self, val_data):
        """
        Validate diffusion model performance on discriminability, unlinkability, and consistency.
        """
        self.gcn.eval()
        self.diffusion.eval()

        same_key_match    = []
        same_key_mismatch = []

        diff_key_match    = []
        diff_key_mismatch = []

        with torch.no_grad():
            for x1, edge_index1, x2, edge_index2, label in val_data:
                Z1 = self.extract_features(x1, edge_index1)
                Z2 = self.extract_features(x2, edge_index2)

                if Z1.dim() == 1:
                    Z1 = Z1.unsqueeze(0)
                if Z2.dim() == 1:
                    Z2 = Z2.unsqueeze(0)
                
                key  = self.get_key()
                Z_T1 = self.protect_template(Z1, key)
                Z_T2 = self.protect_template(Z2, key)

                #sim_same_key = torch.clamp(F.cosine_similarity(Z_T1.unsqueeze(0), Z_T2.unsqueeze(0)), min=0).item()
                #sim_same_key = F.cosine_similarity(Z_T1.unsqueeze(0), Z_T1_diff.unsqueeze(0)).item()
                sim_same_key, _ = pearsonr(Z_T1.cpu(), Z_T2.cpu())

                key_diff = self.get_key()
                while key_diff == key:
                    key_diff = self.get_key()

                Z_T1_diff    = self.protect_template(Z1, key_diff)
                #sim_diff_key = F.cosine_similarity(Z_T1.unsqueeze(0), Z_T1_diff.unsqueeze(0)).item()
                sim_diff_key, _ = pearsonr(Z_T1.cpu(), Z_T1_diff.cpu())

                if label == 1:
                    same_key_match.append(sim_same_key)
                    diff_key_match.append(sim_diff_key)
                else:
                    same_key_mismatch.append(sim_same_key)
                    diff_key_mismatch.append(sim_diff_key)

        match_mean_same_key    = np.mean(same_key_match)    if same_key_match    else 0.0
        mismatch_mean_same_key = np.mean(same_key_mismatch) if same_key_mismatch else 0.0
        match_mean_diff_key    = np.mean(diff_key_match)    if diff_key_match    else 0.0
        mismatch_mean_diff_key = np.mean(diff_key_mismatch) if diff_key_mismatch else 0.0
        
        return {
            'match_mean_same_key'   : match_mean_same_key,
            'mismatch_mean_same_key': mismatch_mean_same_key,
            'match_mean_diff_key'   : match_mean_diff_key,
            'mismatch_mean_diff_key': mismatch_mean_diff_key
        }

    #****************************************************************************
    #****************************************************************************
    def two_stage_training(self, train_data, val_data, test_data,
                         gcn_epochs=30, diffusion_epochs=20, batch_size=32):
        """
        Perform two-stage training: first GCN, then diffusion.
        """
        print("-" * 50)
        # Stage 1: Train GCN
        gcn_losses = self.train_gcn(train_data, val_data, num_epochs=gcn_epochs, batch_size=batch_size)

        print("-" * 50)
        print("-" * 50)
        # Stage 2: Train Diffusion with frozen GCN
        diffusion_losses = self.train_diffusion(train_data, val_data,
                                             num_epochs=diffusion_epochs, batch_size=batch_size)
        print("-" * 50)

        # Evaluate on test set
        print("-" * 50)
        test_metrics, client_similarities, client_labels, lst_entropy, all_similarities_before_diffusion, lst_distance_distributions = self.evaluate(test_data)

        def print_metrics_inline(metrics, title="Test Metrics : "):
            print(f"\n{title}:\n"+ 
                  "\n".join([f"{metric}: {value:.4f}" if not isinstance(value, dict) else 
                             f"{metric}:\n" + "\n".join([f"  {sub_m}: {'N/A' if sub_v is None else f'{sub_v:.4f}'}" 
                                                      for sub_m, sub_v in value.items()]) 
                             for metric, value in metrics.items()]) + 
                  "\n" + "-" * 50)
                  
        print_metrics_inline(test_metrics, title="Test Metrics : ")
        
        return {**gcn_losses, **diffusion_losses}, test_metrics, client_similarities, client_labels, lst_entropy, all_similarities_before_diffusion, lst_distance_distributions
        
               
    def evaluate(self, test_data):
        """
        Evaluate the complete model on test data with comprehensive metrics.
        """
        self.gcn.eval()
        self.diffusion.eval()
        
        # Initialize metrics dictionary
        metrics = {
            'recon_sim': 0.0,             # Average cosine similarity for positive pairs
            'false_accept': 0.0,          # False Accept Rate (FAR)
            'false_reject': 0.0,          # False Reject Rate (FRR)
            'accuracy': 0.0,              # Overall accuracy
            'precision': 0.0,             # Precision
            'recall': 0.0,                # Recall (True Positive Rate)
            'f1_score': 0.0,              # F1 Score
            'auc': 0.0,                   # Area Under ROC Curve
            'eer': 0.0,                   # Equal Error Rate
            'tpr_at_fpr': {              # True Positive Rate at specific False Positive Rates
                '0.1': 0.0,
                '0.01': 0.0,
                '0.001': 0.0
            },
            'verification_time': 0.0,     # Average time for verification
            'reconstruction_fidelity': 0.0, # L2 norm between original and reconstructed
            'hter': 0.0,                  # Half Total Error Rate
            'youden_index': 0.0,          # Youden's J statistic (TPR + TNR - 1)
            'decidability_index': 0.0,    # d-prime or decidability index
        }
        
        # Tracking statistics
        count_pos, count_neg = 0, 0
        correct = 0
        tp, fp, tn, fn = 0, 0, 0, 0
        
        # For ROC/AUC calculations
        all_similarities = []
        all_labels       = []

        all_similarities_before_diffusion = []
        
        lst_entropy      = []

        # For verification time
        total_time = 0.0
        
        # For reconstruction fidelity
        recon_fidelity_sum = 0.0

        intra_class_dist_before = []  # Distances for same subject before diffusion
        inter_class_dist_before = []  # Distances for different subjects before diffusion
        intra_class_dist_after  = []  # Distances for same subject after diffusion
        inter_class_dist_after  = []  # Distances for different subjects after diffusion
  
        with torch.no_grad():
            for x1, edge_index1, x2, edge_index2, label in test_data:
                # Feature extraction
                start_time = time.time()
                #Z1 = self.extract_features(x1, edge_index1)
                #Z2 = self.extract_features(x2, edge_index2)

                #**********************************************
                x1 = x1.to(torch.float32)
                x2 = x2.to(torch.float32)
                Z1 = F.normalize(x1.flatten(), p=2, dim=-1).to(self.device)
                Z2 = F.normalize(x2.flatten(), p=2, dim=-1).to(self.device)
                #**********************************************
                
                # Protection and reconstruction
                key      = self.get_key()

                Z_T1     = self.protect_template(Z1, key)
                Z_recon1 = self.reconstruct_template(Z_T1, key)
                
                Z_T2     = self.protect_template(Z2, key)

                sim = F.cosine_similarity(Z_recon1.unsqueeze(0), Z1.unsqueeze(0)).item()

                # Reconstruction fidelity - L2 distance between original and reconstructed
                recon_fidelity     =  torch.norm(Z1 - Z_recon1).item()
                recon_fidelity_sum += recon_fidelity

                # Similarity calculation
                sim_sys = F.cosine_similarity(Z_T1.unsqueeze(0), Z_T2.unsqueeze(0)).item()

                # Prediction
                pred = 1 if sim_sys > self.threshold else 0
                
                # Track verification time
                end_time   =  time.time()
                total_time += (end_time - start_time)
                
                # Store for ROC/AUC calculation
                all_similarities.append(sim_sys)
                all_labels.append(label)
                
                # Shannon_entropy
                Zi1 = torch.flatten(x1).numpy()
                Zi2 = torch.flatten(x2).numpy()
                entropy_Z1, entropy_Z_T1, mutual_info1, info_loss1, info_preserve1 = evaluate_btp_metrics(Zi1, Z_T1.cpu().numpy())
                entropy_Z2, entropy_Z_T2, mutual_info2, info_loss2, info_preserve2 = evaluate_btp_metrics(Zi2, Z_T2.cpu().numpy())
        
                lst_entropy.append([entropy_Z1, entropy_Z_T1, mutual_info1, info_loss1, info_preserve1])
                lst_entropy.append([entropy_Z2, entropy_Z_T2, mutual_info2, info_loss2, info_preserve2])
                           
                # Update counters
                if pred == label:
                    correct += 1
                
                # Compute cosine distances (1 - cosine similarity)
                sim_before_diffusion  = F.cosine_similarity(Z1.unsqueeze(0), Z2.unsqueeze(0)).item()
                dist_before           = 1 - sim_before_diffusion
                dist_after            = 1 - sim_sys
                 
                all_similarities_before_diffusion.append(sim_before_diffusion)
 
                # Calculate confusion matrix elements
                if label == 1:
                    intra_class_dist_before.append(dist_before)
                    intra_class_dist_after.append(dist_after)

                    metrics['recon_sim'] += sim
                    count_pos += 1
                    if pred == 1:
                        tp += 1
                    else:
                        fn += 1
                        metrics['false_reject'] += 1
                else:
                    inter_class_dist_before.append(dist_before)
                    inter_class_dist_after.append(dist_after)

                    count_neg += 1
                    if pred == 1:
                        fp += 1
                        metrics['false_accept'] += 1
                    else:
                        tn += 1
        
        total = count_pos + count_neg
        
        # Calculate ROC and AUC metrics using sklearn
        if count_pos > 0 and count_neg > 0:  # Only if we have both positive and negative samples
            all_labels       = np.array(all_labels)
            all_similarities = np.array(all_similarities)
            
            try:
                # Calculate ROC curve and AUC
                fpr, tpr, thresholds = roc_curve(all_labels, all_similarities)
                metrics['auc']       = roc_auc_score(all_labels, all_similarities)
                
                # Calculate Equal Error Rate (EER)
                fnr = 1 - tpr
                eer_idx = np.argmin(np.abs(fpr - fnr))
                metrics['eer'] = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
                
                # TPR at specific FPRs
                for target_fpr in ['0.1', '0.01', '0.001']:
                    target = float(target_fpr)
                    if np.max(fpr) >= target:
                        idx = np.argmin(np.abs(fpr - target))
                        metrics['tpr_at_fpr'][target_fpr] = tpr[idx]
                    else:
                        metrics['tpr_at_fpr'][target_fpr] = None  # Not enough data to reach this FPR
                        
                # Calculate d-prime (decidability index)
                if tp > 0 and fp > 0 and tn > 0 and fn > 0:
                    pos_scores = all_similarities[all_labels == 1]
                    neg_scores = all_similarities[all_labels == 0]
                    if len(pos_scores) > 0 and len(neg_scores) > 0:
                        pos_mean = np.mean(pos_scores)
                        neg_mean = np.mean(neg_scores)
                        pos_std  = np.std(pos_scores)
                        neg_std  = np.std(neg_scores)
                        if pos_std > 0 and neg_std > 0:
                            metrics['decidability_index'] = abs(pos_mean - neg_mean) / np.sqrt((pos_std**2 + neg_std**2) / 2)
            except Exception as e:
                print(f"Error calculating ROC/AUC metrics: {e}")
        
        # Calculate basic metrics
        metrics['recon_sim']    /= count_pos       if count_pos > 0 else 1
        metrics['false_accept'] /= count_neg       if count_neg > 0 else 1  # FAR
        metrics['false_reject'] /= count_pos       if count_pos > 0 else 1  # FRR
        metrics['accuracy']     =  correct / total if total > 0     else 0
        
        # Calculate additional metrics
        if tp + fp > 0:
            metrics['precision'] = tp / (tp + fp)
        if tp + fn > 0:
            metrics['recall'] = tp / (tp + fn)
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        
        # Half Total Error Rate (HTER)
        metrics['hter'] = (metrics['false_accept'] + metrics['false_reject']) / 2
        
        # Youden's index (J statistic)
        sensitivity = metrics['recall']
        specificity = 1 - metrics['false_accept']
        metrics['youden_index'] = sensitivity + specificity - 1
        
        # Average verification time
        metrics['verification_time'] = total_time / total if total > 0 else 0
        
        # Average reconstruction fidelity
        metrics['reconstruction_fidelity'] = recon_fidelity_sum / total if total > 0 else 0
        
        return metrics, all_similarities.tolist(), all_labels.tolist(), lst_entropy, all_similarities_before_diffusion, {'intra_class_dist_before': intra_class_dist_before,'inter_class_dist_before': inter_class_dist_before,'intra_class_dist_after': intra_class_dist_after,'inter_class_dist_after': inter_class_dist_after}
        
             
    def evaluate_unlinkability(self, test_data, same_key=False):
        self.gcn.eval()
        self.diffusion.eval()
    
        correlations_match        = []
        correlations_mismatch     = []
        correlations_dif_match    = []
        correlations_dif_mismatch = []
        similaritys_match         = []
        similarities_mismatch     = []
        similaritys_dif_match     = []
        similarities_dif_mismatch = []
    
        with torch.no_grad():
            for x1, edge_index1, x2, edge_index2, label in test_data:
                # Feature extraction
                Z1 = self.extract_features(x1, edge_index1)
                Z2 = self.extract_features(x2, edge_index2)
                
                Z_T1 = None
                Z_T2 = None
                
                if same_key:
                    key  = self.get_key()
                    Z_T1 = self.protect_template(Z1, key)
                    Z_T2 = self.protect_template(Z2, key)
                else:
                    key1 = self.get_key()
                    key2 = self.get_key()
                    while key1 == key2:
                        key1 = self.get_key()
                        key2 = self.get_key()
    
                    Z_T1  = self.protect_template(Z1, key1)
                    Z_T2  = self.protect_template(Z2, key2)
    
        
                # Similarity calculation
                sim_no_dif  = F.cosine_similarity(Z1.unsqueeze(0), Z2.unsqueeze(0)).item()
                sim_dif     = F.cosine_similarity(Z_T1.unsqueeze(0), Z_T2.unsqueeze(0)).item()
                
                # Unlinkability
                corr, _     = pearsonr(Z1.cpu(), Z2.cpu())
                corr_dif, _ = pearsonr(Z_T1.cpu(), Z_T2.cpu())
    
                if label == 1:
                    correlations_match.append(corr)
                    correlations_dif_match.append(corr_dif)
                    similaritys_match.append(sim_no_dif)
                    similaritys_dif_match.append(sim_dif)
                else:
                    correlations_mismatch.append(corr)
                    correlations_dif_mismatch.append(corr_dif)
                    similarities_mismatch.append(sim_no_dif)
                    similarities_dif_mismatch.append(sim_dif)
    
        # Metrics
        mean_corr_match             = np.mean(correlations_match)
        std_corr_match              = np.std(correlations_match)
        t_stat_match, p_value_match = ttest_1samp(correlations_match, 0)
        mean_sim_match              = np.mean(similaritys_match)
        std_sim_match               = np.std(similaritys_match)
    
        mean_corr_mismatch                = np.mean(correlations_mismatch)
        std_corr_mismatch                 = np.std(correlations_mismatch)
        t_stat_mismatch, p_value_mismatch = ttest_1samp(correlations_mismatch, 0)
        mean_sim_mismatch                 = np.mean(similarities_mismatch)
        std_sim_mismatch                  = np.std(similarities_mismatch)
    
        mean_corr_dif_match                 = np.mean(correlations_dif_match)
        std_corr_dif_match                  = np.std(correlations_dif_match)
        t_stat_dif_match, p_value_dif_match = ttest_1samp(correlations_dif_match, 0)
        mean_sim_dif_match                  = np.mean(similaritys_dif_match)
        std_sim_dif_match                   = np.std(similaritys_dif_match)
    
        mean_corr_dif_mismatch                    = np.mean(correlations_dif_mismatch)
        std_corr_dif_mismatch                     = np.std(correlations_dif_mismatch)
        t_stat_dif_mismatch, p_value_dif_mismatch = ttest_1samp(correlations_dif_mismatch, 0)
        mean_sim_dif_mismatch                     = np.mean(similarities_dif_mismatch)
        std_sim_dif_mismatch                      = np.std(similarities_dif_mismatch)
    
        all_metrics  = [mean_corr_match, std_corr_match, t_stat_match, p_value_match, mean_sim_match, std_sim_match,
        mean_corr_mismatch, std_corr_mismatch, t_stat_mismatch, p_value_mismatch, mean_sim_mismatch, std_sim_mismatch,
        mean_corr_dif_match, std_corr_dif_match, t_stat_dif_match, p_value_dif_match, mean_sim_dif_match, std_sim_dif_match,
        mean_corr_dif_mismatch, std_corr_dif_mismatch, t_stat_dif_mismatch, p_value_dif_mismatch, mean_sim_dif_mismatch, std_sim_dif_mismatch]
        
        return all_metrics   

    #****************************************************************************     
    def preimage_attack(self, test_data):
        """
        Simulate a black-box preimage attack and compute security metrics.
        
        Args:
        Returns:
            dict: Metrics including mse, fmr, fnmr, eer, mi, d_unlink, sra.
        """
        self.gcn.eval()
        self.diffusion.eval()

        attack_evaluator = PreimageAttack(self.diffusion)
        lst_similarity_score = []

        with torch.no_grad():
            for x1, edge_index1, x2, edge_index2, label in test_data[0:10]:
                # Feature extraction
                Z = self.extract_features(x1, edge_index1)

                key  = self.get_key()
                Z_T  = self.protect_template(Z, key)

                results = attack_evaluator._run_attack(Z_T, Z_original=Z, max_iterations=500, lr=0.01, threshold=0.95, norm_bound=10.0)
                lst_similarity_score.append(results)

        return lst_similarity_score


      


