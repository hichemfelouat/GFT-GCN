import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy

try:
    import torchmin
    TORCHMIN_AVAILABLE = True
except ImportError:
    TORCHMIN_AVAILABLE = False

print("TORCHMIN_AVAILABLE =", TORCHMIN_AVAILABLE)

if TORCHMIN_AVAILABLE:
    print("torchmin is available!")
else:
    print("torchmin is NOT available.")


class PreimageAttack:
    def __init__(self, diffusion):
        """
        Initialize the preimage attack evaluator for privacy-preserving 3D face mesh recognition.
        """
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = diffusion.to(self.device)

    def get_key(self):
        """
        Generate a user-specific key for diffusion.

        Returns:
            torch.Tensor: Random key vector.
        """
        return torch.randint(0, 1000000, (1,)).item()

    def protect_template(self, Z, key, requires_grad=False):
        """
        Apply spectral diffusion to protect the feature vector.

        Args:
            Z (torch.Tensor)    : Input feature vector.
            key (torch.Tensor)  : User-specific key.
            requires_grad (bool): Whether to track gradients for the output.

        Returns:
            torch.Tensor: Protected template Z_T.
        """

        if requires_grad:
            return self.diffusion(Z, key)
        else:
            with torch.no_grad():
                return self.diffusion(Z, key)

    def black_box_oracle(self, Z_prime, Z_T_target):
        """
        Simulate black-box system returning cosine similarity.

        Args:
            Z_prime (torch.Tensor): Candidate feature vector.
            Z_T_target (torch.Tensor): Protected template.

        Returns:
            torch.Tensor: Cosine similarity score.
        """
        key       = self.get_key()
        Z_prime_T = self.protect_template(Z_prime, key, requires_grad=True)
        return self._cosine_similarity(Z_prime_T, Z_T_target)

    @staticmethod
    def _cosine_similarity(a, b):
        """
        Compute cosine similarity between tensors a and b.

        Args:
            a (torch.Tensor): First tensor.
            b (torch.Tensor): Second tensor.

        Returns:
            torch.Tensor: Cosine similarity.
        """
        return F.cosine_similarity(a.flatten(), b.flatten(), dim=0)

    @staticmethod
    def _estimate_mutual_information(z, z_prime, bins=50):
        """
        Estimate mutual information using histogram-based method.

        Args:
            z (torch.Tensor): Original feature.
            z_prime (torch.Tensor): Reconstructed feature.
            bins (int): Number of histogram bins.

        Returns:
            float: Mutual information estimate.
        """
        z_flat = z.flatten().cpu().numpy()
        z_prime_flat = z_prime.flatten().cpu().numpy()
        hist_2d, _, _ = np.histogram2d(z_flat, z_prime_flat, bins=bins, density=True)
        p_xy = hist_2d / hist_2d.sum()
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        p_xy_flat = np.clip(p_xy.flatten(), 1e-10, None)
        p_x_p_y = np.clip((p_x[:, None] * p_y[None, :]).flatten(), 1e-10, None)
        mi = entropy(p_xy_flat, p_x_p_y)
        return mi

    @staticmethod
    def _compute_mse(z, z_prime):
        """
        Compute Mean Squared Error between original and reconstructed features.

        Args:
            z (torch.Tensor): Original feature.
            z_prime (torch.Tensor): Reconstructed feature.

        Returns:
            float: MSE value.
        """
        return torch.mean((z - z_prime) ** 2).item()

    def _run_attack(self, Z_T, Z_original=None, max_iterations=500, lr=0.01, threshold=0.9, norm_bound=10.0):
        """
        Constrained-Optimized Similarity-Based Attack (CSA) as per Wang et al. (arXiv:2006.13051).
        Optimizes a preimage to maximize similarity with the protected template under an L2 norm constraint.

        Args:
            Z_T (torch.Tensor): Protected template.
            Z_original (torch.Tensor, optional): Original feature for evaluation metrics.
            max_iterations (int): Maximum number of optimization iterations.
            lr (float): Learning rate for the attack.
            threshold (float): Similarity threshold for Attack Success Rate.
            norm_bound (float): L2 norm bound for the preimage.

        Returns:
            dict: Results including best reconstructed feature, similarity score, ASR, MSE, and MI.
        """
        Z_T     = Z_T.to(self.device)
        Z_prime = torch.randn_like(Z_T, device=self.device, requires_grad=True)

        if TORCHMIN_AVAILABLE:
            # Use torchmin for constrained optimization
           #print("We use torchmin for constrained optimization.")
            def objective(Z_prime):
                return -self.black_box_oracle(Z_prime, Z_T)

            """
            constraints = [{'type': 'ineq', 'fun': lambda x: norm_bound**2 - torch.norm(x, p=2)**2}]
            result = torchmin.minimize(
                objective, Z_prime, method='l-bfgs-b', constraints=constraints, max_iter=max_iterations
            )
            """
            constraint = {
                'fun': lambda x: norm_bound**2 - torch.norm(x, p=2)**2,
                'lb': 0
            }

            result = torchmin.minimize_constr(
                objective,
                Z_prime,
                constr=constraint,
                max_iter=max_iterations
            )

            best_Z_prime = result.x
            best_score   = -result.fun.item()
        else:
            # Fallback to unconstrained optimization with penalty term
            #print("We do not use torchmin for constrained optimization.")
            optimizer      = torch.optim.Adam([Z_prime], lr=lr)
            best_Z_prime   = Z_prime.clone().detach()
            best_score     = -float('inf')
            penalty_weight = 100.0  # Weight for L2 norm penalty

            for i in range(max_iterations):
                optimizer.zero_grad()
                sim_score = self.black_box_oracle(Z_prime, Z_T)
                norm_penalty = penalty_weight * F.relu(torch.norm(Z_prime, p=2) - norm_bound)**2
                loss = -sim_score + norm_penalty

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    current_score = self.black_box_oracle(Z_prime, Z_T)
                    if current_score > best_score:
                        best_score = current_score
                        best_Z_prime = Z_prime.clone().detach()

        # Compute evaluation metrics
        results = {
            "similarity_score": best_score,
            "ASR": 1.0 if best_score >= threshold else 0.0
        }

        if Z_original is not None:
            results["MSE"] = self._compute_mse(Z_original.to(self.device), best_Z_prime)
            results["MI"]  = self._estimate_mutual_information(Z_original.to(self.device), best_Z_prime)

        return results


