import torch
import torch.nn.functional as F
from torch import nn, optim

# -------------------------
# Online KMeans (PyTorch)
# -------------------------
class OnlineKMeans:
    def __init__(self, n_clusters, feat_dim, device='cpu', init='random'):
        """
        n_clusters: K
        feat_dim: dimensionality of features
        device: 'cpu' or 'cuda'
        init: 'random' or 'kmeans++' (we implement random init)
        """
        self.K = n_clusters
        self.dim = feat_dim
        self.device = device
        self.centroids = torch.zeros(self.K, self.dim, device=self.device)
        self.counts = torch.zeros(self.K, device=self.device)  # number of points assigned to cluster
        self.inited = False
        self.eps = 1e-6

        self.init = init

    def initialize(self, init_feats):
        """
        init_feats: tensor (N, dim) used to initialize centroids. picks K random points.
        """
        N = init_feats.shape[0]
        if N < self.K:
            # fallback: random normal
            self.centroids = torch.randn(self.K, self.dim, device=self.device)
        else:
            idx = torch.randperm(N, device=init_feats.device)[:self.K]
            self.centroids = init_feats[idx].to(self.device).clone()
        self.counts = torch.zeros(self.K, device=self.device)
        self.inited = True

    @torch.no_grad()
    def partial_fit(self, feats):
        """
        feats: (B, dim) tensor on device
        Returns:
            labels: (B,) long tensor with cluster indices for each feat
        Updates centroids and counts incrementally using exact average:
            new_centroid = (count_k * centroid_k + sum_assigned_k) / (count_k + n_assigned_k)
        """
        if not self.inited:
            self.initialize(feats)

        B = feats.shape[0]
        # compute squared distances (B, K)
        # dist = ||x - c||^2 = x^2 + c^2 - 2 x.c
        x2 = (feats * feats).sum(dim=1, keepdim=True)           # (B,1)
        c2 = (self.centroids * self.centroids).sum(dim=1).unsqueeze(0)  # (1,K)
        xc = feats @ self.centroids.t()                        # (B,K)
        dists = x2 + c2 - 2.0 * xc                             # (B,K)

        labels = torch.argmin(dists, dim=1)  # (B,)

        # aggregate sums per cluster
        # compute sum_assigned_k and n_assigned_k
        # We'll do scatter_add
        sums = torch.zeros_like(self.centroids)  # (K, dim)
        counts_new = torch.zeros(self.K, device=self.device)
        # convert to float for sums
        labels_long = labels.to(torch.long)
        sums = sums.index_add(0, labels_long, feats)
        counts_new = counts_new.index_add(0, labels_long, torch.ones(B, device=self.device))

        # update centroids with counts
        # For clusters with new points:
        nonzero = counts_new > 0
        if nonzero.any():
            # new_count = counts + counts_new
            new_counts = self.counts.clone()
            new_counts[nonzero] += counts_new[nonzero]

            # updated centroid = (old_count*old_centroid + sum_new) / new_count
            old_contrib = (self.counts[nonzero].unsqueeze(1) * self.centroids[nonzero])
            new_centroid_vals = (old_contrib + sums[nonzero]) / (new_counts[nonzero].unsqueeze(1) + self.eps)
            self.centroids[nonzero] = new_centroid_vals
            self.counts[nonzero] = new_counts[nonzero]

        return labels


# # -------------------------
# # Discriminator training step (per mini-batch)
# # -------------------------
# def train_discriminator_on_minibatch(disc_model, disc_opt, features, online_kmeans, device):
#     """
#     features: (B, feat_dim) torch tensor (detached from encoder if required)
#     online_kmeans: instance of OnlineKMeans
#     disc_model: torch.nn.Module that takes features and returns logits (B, K)
#     disc_opt: optimizer for discriminator
#     """
#     # 1) Get cluster labels (and update centroids)
#     with torch.no_grad():
#         labels = online_kmeans.partial_fit(features)  # (B,) long

#     # 2) Compute logits from discriminator (make sure features detached so encoder not updated)
#     # If your discriminator uses its own feature head, use that; here we assume features are ready.
#     logits = disc_model(features)  # (B, K)

#     # 3) Cross-entropy loss
#     loss = F.cross_entropy(logits, labels)

#     # 4) update discriminator params only
#     disc_opt.zero_grad()
#     loss.backward()
#     disc_opt.step()

#     # 5) compute intrinsic reward (log p(z|x) - log(1/K)) for this minibatch (detached)
#     with torch.no_grad():
#         log_probs = F.log_softmax(logits, dim=-1)
#         log_p_correct = log_probs[torch.arange(labels.shape[0], device=labels.device), labels]
#         log_prior = -torch.log(torch.tensor(float(online_kmeans.K), device=labels.device))
#         intrinsic_reward = log_p_correct - log_prior  # shape (B,)
#     return loss.item(), labels, logits.detach(), intrinsic_reward.detach()


# # -------------------------
# # Example integration with your prepare_batch
# # -------------------------
# # Suppose `batch_trajs` is the output of prepare_batch(...)
# # features = batch_trajs["features"] shape (N, feat_dim)
# # We'll process it in minibatches to update discriminator and online kmeans

# def discriminator_training_epoch(agent, disc_opt, online_kmeans, batch_trajs, device, mb_size=1024):
#     """
#     agent: your agent object, must expose `discriminator` that accepts features -> logits
#     disc_opt: optimizer for discriminator parameters
#     online_kmeans: instance of OnlineKMeans
#     batch_trajs: prepared batch dict (all flattened as you prepared)
#     """
#     features_all = batch_trajs["features"].to(device)  # (N, feat_dim)
#     N = features_all.shape[0]
#     idxs = torch.randperm(N, device=device)

#     total_loss = 0.0
#     all_labels = []
#     all_intrinsic = []

#     for start in range(0, N, mb_size):
#         end = min(start + mb_size, N)
#         mb_idx = idxs[start:end]
#         feats_mb = features_all[mb_idx]
#         # IMPORTANT: detach features so discriminator update doesn't backprop to encoder
#         feats_mb = feats_mb.detach()

#         loss_val, labels, logits, intrinsic = train_discriminator_on_minibatch(
#             disc_model=agent.discriminator,
#             disc_opt=disc_opt,
#             features=feats_mb,
#             online_kmeans=online_kmeans,
#             device=device
#         )
#         total_loss += loss_val * (end - start)
#         all_labels.append(labels.cpu())
#         all_intrinsic.append(intrinsic.cpu())

#     avg_loss = total_loss / N
#     all_labels = torch.cat(all_labels, dim=0)
#     all_intrinsic = torch.cat(all_intrinsic, dim=0)
#     return avg_loss, all_labels, all_intrinsic
