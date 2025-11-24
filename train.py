import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

import time
import os
import random

from scripts.gcn import*
from scripts.diffusion import*
from scripts.client import*
from scripts.server import*

from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score, f1_score
import pandas as pd
from scipy.stats import entropy, pearsonr, ttest_1samp

os.system("clear")

import numpy as np
from sklearn.metrics import roc_curve

def evaluate_metrics_attack(similarities, labels, threshold_range=(0.0, 1.0, 0.01)):
    """
    Compute evaluation metrics and find the best threshold for Attack Success Rate (ASR).
    Handles case where labels are all 1 (true matches) and avoids division by zero.

    Args:
        similarities (list): List of cosine similarity scores from the preimage attack.
        labels (list): List of binary labels (all 1 for true matches).
        threshold_range (tuple): (min, max, step) for threshold search.

    Returns:
        dict: Metrics including ASR, best threshold, and best ASR.
    """
    # Convert inputs to numpy arrays
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # Validate inputs
    if not np.all(labels == 1):
        raise ValueError("All labels must be 1 for true matches.")
    
    # Initialize results
    results = {
        "ASR": [],
        "thresholds": [],
        "best_threshold": None,
        "best_ASR": None
    }
    
    # Compute ASR for a range of thresholds
    threshold_values = np.arange(*threshold_range)
    asr_values = []
    
    for thresh in threshold_values:
        asr = (similarities >= thresh).mean()  # Proportion of similarities >= threshold
        asr_values.append(asr)
    
    results["ASR"] = asr_values
    results["thresholds"] = threshold_values.tolist()

    # Find best threshold for maximum ASR
    if asr_values:
        best_index = np.argmax(asr_values)
        results["best_threshold"] = threshold_values[best_index]
        results["best_ASR"] = asr_values[best_index]
    else:
        results["best_threshold"] = 0.0
        results["best_ASR"] = 0.0
    
    return results

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def augment_feature_dataset(train_data, noise_std=0.05, scale_range=(0.95, 1.05), dropout_prob=0.1, num_augmentations=1):
    """
    Apply feature augmentation to the training dataset for the GFT-GCN system, with optional oversampling.
    
    Parameters:
    - train_data (list): List of tuples (x1, edge_index1, x2, edge_index2, label), where:
        - x1, x2: Feature vectors (torch.Tensor, e.g., GFT coefficients or GCN output).
        - edge_index1, edge_index2: Edge indices (np.ndarray, shape (m, 2)) for graph connectivity.
        - label: Subject ID (int or str).
    - noise_std (float): Standard deviation for Gaussian noise on features.
    - scale_range (tuple): (min, max) scaling factors for feature magnitude.
    - dropout_prob (float): Probability of dropping a feature dimension.
    - num_augmentations (int): Number of augmented versions to generate per sample.
    
    Returns:
    - train_data_aug (list): Augmented training data with (x1_aug, edge_index1, x2_aug, edge_index2, label).
    """
    def augment_features(features, noise_std, scale_range, dropout_prob):
        """Augment feature vector."""
        features_aug = features.clone()
        
        # Ensure features is 2D for consistent processing
        if features_aug.dim() == 1:
            features_aug = features_aug.view(1, -1)
        
        # 1. Spectral Noise
        if noise_std > 0:
            noise = torch.normal(0, noise_std, features_aug.shape, device=features_aug.device)
            features_aug = features_aug + noise
        
        # 2. Feature Scaling
        scale_factor = torch.FloatTensor(1).uniform_(scale_range[0], scale_range[1]).to(features_aug.device)
        features_aug = features_aug * scale_factor
        
        # 3. Feature Dropout
        if dropout_prob > 0:
            mask = torch.binomial(torch.ones_like(features_aug), torch.tensor(1 - dropout_prob, dtype=features_aug.dtype, device=features_aug.device))
            features_aug = features_aug * mask
        
        # Reshape back to original shape
        if features.dim() == 1:
            features_aug = features_aug.flatten()
        
        return features_aug

    # Augment training data
    train_data_aug = []
    for i, (x1, edge_index1, x2, edge_index2, label) in enumerate(train_data):
        # Generate multiple augmented versions
        for _ in range(num_augmentations):
            # Augment both x1 and x2 feature vectors
            x1_aug = augment_features(x1, noise_std, scale_range, dropout_prob)
            x2_aug = augment_features(x2, noise_std, scale_range, dropout_prob)
            # Preserve edge indices and label
            train_data_aug.append((x1_aug, edge_index1, x2_aug, edge_index2, label))
    
    return train_data_aug
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def get_same_user(test_data):
  for x1, edge_index1, x2, edge_index2, label in test_data:
    if label == 1:
      return x1, edge_index1, x2, edge_index2
  return None, None, None, None

def get_diff_user(test_data):
  for x1, edge_index1, x2, edge_index2, label in test_data:
    if label == 0:
      return x1, edge_index1, x2, edge_index2
  return None, None, None, None

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def evaluate_facial_recognition(all_similarities, all_labels, thresholds=None):
    """
    Evaluate a facial recognition system using multiple metrics.

    Parameters:
    -----------
    all_similarities : list of lists. Each sublist contains similarity scores for one client.
    all_labels : list of lists. Each sublist contains corresponding binary labels (1 for match, 0 for non-match).
    thresholds : list or numpy array, optional
        Thresholds to evaluate for determining best operating point.
        If None, uses np.arange(0.01, 1.00, 0.01).
    """

    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    # Flatten all similarities and labels for global evaluation
    flat_similarities = np.concatenate([np.array(client_sim) for client_sim in all_similarities])
    flat_labels       = np.concatenate([np.array(client_lab) for client_lab in all_labels])

    # Calculate ROC curve and AUC
    fpr, tpr, roc_thresholds = roc_curve(flat_labels, flat_similarities)
    roc_auc                  = auc(fpr, tpr)

    # Calculate Precision-Recall curve and AUC
    precision, recall, pr_thresholds = precision_recall_curve(flat_labels, flat_similarities)
    pr_auc                           = auc(recall, precision)

    # Calculate metrics at different thresholds
    threshold_metrics = []

    for threshold in thresholds:
        predictions   = (flat_similarities >= threshold).astype(int)
        # True positive rate (TPR) / Recall
        tpr_val       = np.sum((predictions == 1) & (flat_labels == 1)) / np.sum(flat_labels == 1) if np.sum(flat_labels == 1) > 0 else 0
        # False positive rate (FPR)
        fpr_val       = np.sum((predictions == 1) & (flat_labels == 0)) / np.sum(flat_labels == 0) if np.sum(flat_labels == 0) > 0 else 0
        # Precision (PPV)
        precision_val = np.sum((predictions == 1) & (flat_labels == 1)) / np.sum(predictions == 1) if np.sum(predictions == 1) > 0 else 0
        # Accuracy
        accuracy      = accuracy_score(flat_labels, predictions)
        # F1 score
        f1            = f1_score(flat_labels, predictions, zero_division=0)

        threshold_metrics.append({
            'threshold': threshold,
            'tpr': tpr_val,
            'fpr': fpr_val,
            'precision': precision_val,
            'accuracy': accuracy,
            'f1': f1
        })

    # Find best threshold based on F1 score
    threshold_df       = pd.DataFrame(threshold_metrics)
    best_threshold_idx = threshold_df['f1'].idxmax()
    best_threshold     = threshold_df.loc[best_threshold_idx, 'threshold']

    # Calculate metrics per client
    client_metrics = []
    for i, (client_sim, client_lab) in enumerate(zip(all_similarities, all_labels)):
        client_sim = np.array(client_sim)
        client_lab = np.array(client_lab)

        # Skip if client has no data
        if len(client_sim) == 0:
            continue

        # Calculate client-specific metrics
        client_pred     = (client_sim >= best_threshold).astype(int)
        # Client-specific TPR
        client_tpr      = np.sum((client_pred == 1) & (client_lab == 1)) / np.sum(client_lab == 1) if np.sum(client_lab == 1) > 0 else 0
        # Client-specific FPR
        client_fpr      = np.sum((client_pred == 1) & (client_lab == 0)) / np.sum(client_lab == 0) if np.sum(client_lab == 0) > 0 else 0
        # Client-specific accuracy
        client_accuracy = accuracy_score(client_lab, client_pred)
        # Client-specific F1
        client_f1       = f1_score(client_lab, client_pred, zero_division=0)

        client_metrics.append({
            'client_id': i,
            'samples': len(client_sim),
            'tpr': client_tpr,
            'fpr': client_fpr,
            'accuracy': client_accuracy,
            'f1': client_f1
        })

    # Calculate Equal Error Rate (EER)
    diff_rates   = np.abs(fpr - (1 - tpr))
    min_diff_idx = np.argmin(diff_rates)
    eer          = (fpr[min_diff_idx] + (1 - tpr[min_diff_idx])) / 2

    # Calculate metrics at best threshold
    best_predictions = (flat_similarities >= best_threshold).astype(int)
    best_accuracy    = accuracy_score(flat_labels, best_predictions)
    best_f1          = f1_score(flat_labels, best_predictions, zero_division=0)

    # Return results
    results = {
        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
        'roc_auc': roc_auc,
        'pr_curve': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds},
        'pr_auc': pr_auc,
        'threshold_metrics': threshold_metrics,
        'best_threshold': best_threshold,
        'best_accuracy': best_accuracy,
        'best_f1': best_f1,
        'eer': eer,
        'client_metrics': client_metrics
    }

    # 1. Evaluate the facial recognition system
    print("# Evaluating facial recognition system...")
    # 2. Print summary statistics
    print("## Summary Statistics:")
    print(f"- ROC AUC                   : {results['roc_auc']:.4f}")
    print(f"- PR AUC                    : {results['pr_auc']:.4f}")
    print(f"- Equal Error Rate (EER)    : {results['eer']:.4f}")
    print(f"- Best threshold (by F1)    : {results['best_threshold']:.4f}")
    print(f"- Accuracy at best threshold: {results['best_accuracy']:.4f}")
    print(f"- F1 score at best threshold: {results['best_f1']:.4f}")

    print(f"\n## Client Performance Summary:")
    print(f"- Total clients evaluated: {len(results['client_metrics'])}")
    print(f"- Average client TPR     : {np.mean([client['tpr'] for client in results['client_metrics']]):.4f}")
    print(f"- Average client FPR     : {np.mean([client['fpr'] for client in results['client_metrics']]):.4f}")
    print(f"- Average client accuracy: {np.mean([client['accuracy'] for client in results['client_metrics']]):.4f}")
    print(f"- Average client F1 score: {np.mean([client['f1'] for client in results['client_metrics']]):.4f}")

    print(f"\n## Recommendations")
    print(f"- The system performs best at a threshold of {results['best_threshold']:.4f}.")
    print(f"- This threshold balances false positives and false negatives well.")

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Usage 
def main():
    start_time = time.time()
    print("Starting Privacy-Preserving 3D Face Mesh Recognition Example with Two-Stage Training...")

    dataset_name    = "BU_3DFE" # BU_3DFE FaceScape 
    nbr_features    = 10
    k               = 25
    out_channels    = nbr_features * k
    threshold       = 0.97
    
    diffusion_steps = 50
    
    nbr_client      = 100

    gcn_epochs      = 40
    diffu_epochs    = 40
    
    print(f"k: = {k}, threshold: = {threshold}, diffusion_steps: = {diffusion_steps}, nbr_client: = {nbr_client}, gcn_epochs: = {gcn_epochs}, diffu_epochs: = {diffu_epochs}")

    server               = Server()
    all_similarities     = [] 
    all_labels           = []
    all_client_entropy   = [] 

    all_similarities_before_diffusion = []
    
    all_unlinkability_metrics_samekey = []
    all_unlinkability_metrics_difrkey = []

    all_client_dis_distributions      = []

    all_attack_metrics                = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    for i in range(nbr_client):  
        print("-" * 50)
        print("-" * 50)
        print("Client : ",i)
    
        #***********************************************************************
        # Load data with train/val/test split
        train_data = torch.load("../preprocessed_dataset/"+str(dataset_name)+"_k"+str(k)+"_c100/train_data_"+str(i)+".pth")
        val_data   = torch.load("../preprocessed_dataset/"+str(dataset_name)+"_k"+str(k)+"_c100/val_data_"+str(i)+".pth")
        test_data  = torch.load("../preprocessed_dataset/"+str(dataset_name)+"_k"+str(k)+"_c100/test_data_"+str(i)+".pth")
        
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)} pairs")

        """
        print("One Example : ")
        print("train_data : ",train_data[0][0].shape, type(train_data[0][0]))
        print("train_data : ",train_data[0][1].shape, type(train_data[0][1]))
        print("train_data : ",train_data[0][2].shape, type(train_data[0][2]))
        print("train_data : ",train_data[0][3].shape, type(train_data[0][3]))
        print("train_data : ",train_data[0][4], type(train_data[0][4]))
        """

        # Apply augmentation
        """
        train_data_aug = augment_feature_dataset(
                                                train_data,
                                                noise_std=0.05,  # Spectral noise
                                                scale_range=(0.95, 1.05),  # Feature scaling
                                                dropout_prob=0.1,  # Feature dropout
                                                num_augmentations=3
                                                )
        print(f"train_data_aug: {len(train_data_aug)} pairs")
        """
        
        #***********************************************************************  
        client = Client(in_channels=nbr_features, hidden_channels=128, out_channels=out_channels, 
                        diffusion_steps=diffusion_steps, threshold=threshold).to(device)
        
        # Two-stage training 
        print("Two-stage training started...")
        losses, test_metrics, client_similarities, client_labels, client_lst_entropy, client_similarities_before_diffusion, client_dis_distributions = client.two_stage_training(
            train_data, val_data, test_data,
            gcn_epochs=gcn_epochs, diffusion_epochs=diffu_epochs)
        
        # For evaluation
        all_similarities.append(client_similarities)
        all_labels.append(client_labels)

        all_similarities_before_diffusion.append(client_similarities_before_diffusion)
        
        all_client_entropy = all_client_entropy + client_lst_entropy

        all_client_dis_distributions.append(client_dis_distributions)
        
        # Evaluate Unlinkability
        unlinkability_metrics_samekey = client.evaluate_unlinkability(test_data, same_key=True)
        unlinkability_metrics_difrkey = client.evaluate_unlinkability(test_data, same_key=False)
        all_unlinkability_metrics_samekey.append(unlinkability_metrics_samekey)
        all_unlinkability_metrics_difrkey.append(unlinkability_metrics_difrkey)
        
        # Pre-image attack
        #attack_metrics     = client.preimage_attack(test_data)
        #all_attack_metrics = all_attack_metrics + attack_metrics
        
        #***********************************************************************
        print("-" * 50) 
        # Enrollment
        user_id = "user_"+str(i)
        key     = client.get_key()

        xs1,es1,xs2,es2 = get_same_user(test_data)
        x_enroll = xs1
        e_enroll = es1
        Z_T_enrolled = client.process_mesh(x_enroll, e_enroll, key, is_enrollment=True)
        server.enroll(user_id, Z_T_enrolled)
        
        # Authentication (same user)
        start_time_inf = time.time()
        x_query = xs2
        e_query = es2
        Z_T_query  = client.process_mesh(x_query, e_query, key, is_enrollment=True)
        match_result, similarity = server.match(Z_T_query, user_id, threshold)
        print(f"Authentication (same user, Key {key}):         Match={match_result}, Similarity={similarity:.4f}")
        end_time_inf = time.time()
        print("The execution inf time (in seconds) is : ",end_time_inf-start_time_inf)

        # Authentication (different user)
        xd1,ed1,xd2,ed2 = get_diff_user(test_data) 
        x_query_diff = xd2
        e_query_diff = ed2
        Z_T_diff = client.process_mesh(x_query_diff, e_query_diff, key, is_enrollment=True)
        match_result, similarity = server.match(Z_T_diff, user_id, threshold)
        print(f"Authentication (different user, Key {key}):    Match={match_result}, Similarity={similarity:.4f}")

        # Renewability test
        new_key = client.get_key()
        print(f"New user with Key {new_key} for renewability test")
        
        Z_T_new  = client.process_mesh(x_enroll, e_enroll, new_key, is_enrollment=True)
        match_result, similarity = server.match(Z_T_new, user_id, threshold)
        print(f"Renewability test (new Key {new_key} vs original): Match={match_result}, Similarity={similarity:.4f}")
        print("-" * 50)

        #***********************************************************************
        """
        print("-" * 50)
        j = 0
        for x1, e1, x2, e2, label in test_data:
            print(j)
            j = j + 1
            print("label : ",label)
            keyq = client.get_key()
            x_query = x2
            e_query = e2
            Z_T_query  = client.process_mesh(x_query, e_query, keyq, is_enrollment=True)
            match_result, similarity = server.match(Z_T_query, user_id, threshold)
            print(f"Authentication (same user, Key {keyq}): Match={match_result}, Similarity={similarity:.4f}")   
            print("-" * 50)
        
        """

    #*************************************************************************** 
    print("-" * 50)
    print("# Entropy results:")
    entropy_Z, entropy_Z_T, mutual_info, info_loss, info_preserve = 0,0,0,0,0
    for lst_entropy in all_client_entropy:
        entropy_Z     = entropy_Z     + lst_entropy[0]
        entropy_Z_T   = entropy_Z_T   + lst_entropy[1] 
        mutual_info   = mutual_info   + lst_entropy[2]
        info_loss     = info_loss     + lst_entropy[3] 
        info_preserve = info_preserve + lst_entropy[4]
    
    len_lst       = len(all_client_entropy)
    entropy_Z     = entropy_Z / len_lst
    entropy_Z_T   = entropy_Z_T / len_lst
    mutual_info   = mutual_info / len_lst
    info_loss     = info_loss / len_lst 
    info_preserve = info_preserve / len_lst

    # Print metrics
    print(f"Shannon Entropy of Z    : {entropy_Z:.4f} bits")
    print(f"Shannon Entropy of Z_T  : {entropy_Z_T:.4f} bits")
    print(f"Mutual Information      : {mutual_info:.4f} bits")
    print(f"Information Loss        : {info_loss:.4f}")
    print(f"Information Preservation: {info_preserve:.4f}")
    
    print("-" * 50)
    print("-" * 50)
    print("#Before Diffusion:")
    evaluate_facial_recognition(all_similarities_before_diffusion, all_labels, thresholds=None)

    print("-" * 50)
    print("#After Diffusion:")
    evaluate_facial_recognition(all_similarities, all_labels, thresholds=None)
    
    print("-" * 50)
    print("-" * 50)
    metric_names = ['mean_corr_match', 'std_corr_match', 't_stat_match', 'p_value_match', 'mean_sim_match', 'std_sim_match',
    'mean_corr_mismatch', 'std_corr_mismatch', 't_stat_mismatch', 'p_value_mismatch', 'mean_sim_mismatch', 'std_sim_mismatch',
    'mean_corr_dif_match', 'std_corr_dif_match', 't_stat_dif_match', 'p_value_dif_match', 'mean_sim_dif_match', 'std_sim_dif_match',
    'mean_corr_dif_mismatch', 'std_corr_dif_mismatch', 't_stat_dif_mismatch', 'p_value_dif_mismatch', 'mean_sim_dif_mismatch', 'std_sim_dif_mismatch']

    print("# Unlinkability metrics:")
    print("## Before Applying Diffusion:")
    means_1 = np.mean(all_unlinkability_metrics_samekey, axis=0)
    lin = 0
    for name, mean_value in zip(metric_names, means_1):
        print(f"{name}: {mean_value}")
        lin = lin + 1
        if lin % 6 == 0:
            print("\n")
        if lin >= 12:
            break
    
    print("## The same key:")
    lin = 0
    for name, mean_value in zip(metric_names, means_1):
        if lin >= 12 :
            print(f"{name}: {mean_value}")
            if lin == 17 :
                print("\n")
        lin = lin + 1
         
    print("\n## Different key:")
    means_2 = np.mean(all_unlinkability_metrics_difrkey, axis=0)
    lin     = 0
    for name, mean_value in zip(metric_names, means_2):
        if lin >= 12 :
            print(f"{name}: {mean_value}")
            if lin == 17 :
                print("\n")
        lin = lin + 1
    
    
    #*************************************************************************** 
    """
    all_similarities = np.array(all_similarities, dtype=object)
    all_labels       = np.array(all_labels, dtype=object)
    all_similarities_before_diffusion = np.array(all_similarities_before_diffusion, dtype=object)

    np.save('../metrics/'+str(dataset_name)+'_k'+str(k)+'_c10/all_similarities_k'+str(k)+'_c10_'+str(diffusion_steps)+'.npy', all_similarities)
    np.save('../metrics/'+str(dataset_name)+'_k'+str(k)+'_c10/all_labels_k'+str(k)+'_c10_'+str(diffusion_steps)+'.npy', all_labels)
    np.save('../metrics/'+str(dataset_name)+'_k'+str(k)+'_c10/all_similarities_before_diffusion_k'+str(k)+'_c10_'+str(diffusion_steps)+'.npy', all_similarities_before_diffusion)

    all_client_dis_distributions = np.array(all_client_dis_distributions, dtype=object)
    np.save('../metrics/'+str(dataset_name)+'_k'+str(k)+'_c10/all_dis_distributions_k'+str(k)+'_c10_'+str(diffusion_steps)+'.npy', all_client_dis_distributions)
    """
    #***************************************************************************
     
    print("-" * 50)
    """
    print("Compute pre-image attack metrics:")
    asr_mean = sum(d["ASR"] for d in all_attack_metrics) / len(all_attack_metrics)
    mse_mean = sum(d["MSE"] for d in all_attack_metrics) / len(all_attack_metrics)
    mi_mean  = sum(d["MI"]  for d in all_attack_metrics)  / len(all_attack_metrics)

    

    similarity_score_attack  = []
    for rst in all_attack_metrics:
        similarity_score_attack.append(rst["similarity_score"])
    
    labels_attack = [1]*len(similarity_score_attack)
    # Evaluate metrics
    results = evaluate_metrics_attack(similarity_score_attack, labels_attack)
    
    # Print results
    print(f"Evaluation Metrics:")
    print(f"Mean ASR: {asr_mean:.4f}")
    print(f"Mean MSE: {mse_mean:.4f}")
    print(f"Mean MI : {mi_mean:.4f}")
    mean_sim_att = sum(similarity_score_attack) / len(similarity_score_attack)
    print(f"Mean Sim: {mean_sim_att:.4f}")

 
    print(f"Best Threshold: {results['best_threshold']:.4f}")
    print(f"Best ASR: {results['best_ASR']:.4f}")

    print("-" * 20)
    for t in range(len(results['thresholds'])):
        print(f"{t} - thresholds : {results['thresholds'][t]:.4f}, ASR : {results['ASR'][t]:.4f}")

    """

    #*************************************************************************** 
    print("-" * 50)
    print("-" * 50)
    print("Example completed successfully.")
    end_time = time.time()
    print("The execution time (in seconds) is : ",end_time-start_time)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

