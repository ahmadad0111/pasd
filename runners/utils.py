import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
def compute_advantage(episode_traj, config):
    with torch.no_grad():
        lastgaelam = 0
        for t in reversed(range(config.training.rollout_steps)):
            if t == config.training.rollout_steps - 1:
                nextnonterminal = 0
                nextvalues = 0
            else:
                nextnonterminal = 1.0 - episode_traj["dones"][t + 1]
                nextvalues = episode_traj["values"][t + 1]

            delta = episode_traj["rewards"][t] + config.training.gamma * nextvalues * nextnonterminal - episode_traj["values"][t]
            episode_traj["advantages"][t] = lastgaelam = delta + config.training.gamma * config.training.gae_lambda * nextnonterminal * lastgaelam
        episode_traj["returns"] = episode_traj["advantages"] + episode_traj["values"]
    return episode_traj

def compute_advantage_pop(episode_traj, config):
    with torch.no_grad():
        for agent_ind in range(config.training.population_size):
            lastgaelam_p0 = 0
            lastgaelam_p1 = 0
            for t in reversed(range(config.training.rollout_steps)):
                if t == config.training.rollout_steps - 1:
                    nextnonterminal = 0
                    nextvalues_p0 = 0
                    nextvalues_p1 = 0
                else:
                    nextnonterminal = 1.0 - episode_traj["dones"][agent_ind,t + 1]
                    nextvalues_p0 = episode_traj["p0"]["values"][agent_ind,t + 1]
                    nextvalues_p1 = episode_traj["p1"]["values"][agent_ind,t + 1]
                delta_p0 = episode_traj["rewards"][agent_ind, t] + config.training.gamma * nextvalues_p0 * nextnonterminal - episode_traj["p0"]["values"][agent_ind, t]
                episode_traj["p0"]["advantages"][agent_ind, t] = lastgaelam_p0 = delta_p0 + config.training.gamma * config.training.gae_lambda * nextnonterminal * lastgaelam_p0
                delta_p1 = episode_traj["rewards"][agent_ind, t] + config.training.gamma * nextvalues_p1 * nextnonterminal - episode_traj["p1"]["values"][agent_ind, t]
                episode_traj["p1"]["advantages"][agent_ind, t] = lastgaelam_p1 = delta_p1 + config.training.gamma * config.training.gae_lambda * nextnonterminal * lastgaelam_p1
        episode_traj["p0"]["returns"] = episode_traj["p0"]["advantages"] + episode_traj["p0"]["values"]
        episode_traj["p1"]["returns"] = episode_traj["p1"]["advantages"] + episode_traj["p1"]["values"]
    return episode_traj

def compute_advantage_hi(episode_traj, config):
    with torch.no_grad():
        for ind in range(config.training.num_envs):
            lastgaelam_z = 0
            for t in reversed(range(episode_traj[ind]["hi_rewards"].shape[0])):
                if t == episode_traj[ind]["hi_rewards"].shape[0] - 1:
                    nextnonterminal = 0.0
                    nextvalues_z = 0.0
                else:
                    nextnonterminal = 1.0
                    nextvalues_z = episode_traj[ind]["z_values"][t + 1]
                delta_z = episode_traj[ind]["hi_rewards"][t] + config.training.gamma * nextvalues_z * nextnonterminal - episode_traj[ind]["z_values"][t]
                episode_traj[ind]["z_adv"][t] = lastgaelam_z = delta_z + config.training.gamma * config.training.gae_lambda * nextnonterminal * lastgaelam_z
            episode_traj[ind]["z_ret"] = episode_traj[ind]["z_adv"] + episode_traj[ind]["z_values"]
    return episode_traj


def get_aligned_hi_values(hi_ep_dict, episode_dict, config):
    num_envs = config.training.num_envs
    rollout_steps = config.training.rollout_steps

    # Initialize aligned_hi_values tensor with zeros
    aligned_hi_values = torch.zeros((rollout_steps, num_envs), device=episode_dict["values"].device)

    for env in range(num_envs):
        hi_vals = hi_ep_dict[env]["z_values"]  # shape: [num_option_switches_for_env]
        option_switch_steps = hi_ep_dict[env]["inds"]  # steps at which options switched

        # Forward-fill hi values between option switch steps
        prev_step = 0
        for idx, switch_step in enumerate(option_switch_steps):
            hi_val = hi_vals[idx]  # corresponding hi_value at this switch
            # Fill from prev_step (inclusive) to switch_step (exclusive)
            aligned_hi_values[prev_step:switch_step, env] = hi_val
            prev_step = switch_step
        # Fill remaining steps after last switch
        if prev_step < rollout_steps:
            aligned_hi_values[prev_step:, env] = hi_vals[-1]

    return aligned_hi_values


def compute_advantage_term(episode_dict, hi_ep_dict, config):
    with torch.no_grad():
        aligned_hi_values = get_aligned_hi_values(hi_ep_dict, episode_dict, config)

        # Normalize hi and lo values BEFORE computing termination reward
        all_values = torch.cat([aligned_hi_values, episode_dict["values"]], dim=0)
        mean = all_values.mean()
        std = all_values.std(unbiased=False) + 1e-8

        norm_hi_values = (aligned_hi_values - mean) / std
        norm_lo_values = (episode_dict["values"] - mean) / std

        # Compute normalized termination reward
        term_reward = norm_hi_values - norm_lo_values  # shape: [T, N]
        print(f"Term_reward: {term_reward}")

        term_adv = torch.zeros_like(episode_dict["termination_advantages"])
        lastgaelam_term = torch.zeros(config.training.num_envs, device=episode_dict["values"].device)

        for t in reversed(range(config.training.rollout_steps)):
            if t == config.training.rollout_steps - 1:
                nextnonterminal = torch.zeros(config.training.num_envs, device=episode_dict["values"].device)
                nextvalues = torch.zeros(config.training.num_envs, device=episode_dict["values"].device)
            else:
                nextnonterminal = 1.0 - episode_dict["dones"][t + 1]
                nextvalues = episode_dict["termination_values"][t + 1]

            delta = (
                term_reward[t] +
                config.training.gamma * nextvalues * nextnonterminal -
                episode_dict["termination_values"][t]
            ) 

            lastgaelam_term = (
                delta +
                config.training.gamma * config.training.gae_lambda * nextnonterminal * lastgaelam_term
            )
            term_adv[t] = lastgaelam_term

        episode_dict["termination_advantages"] = term_adv
        episode_dict["termination_returns"] = term_adv + episode_dict["termination_values"]

    return episode_dict


def compute_termination_advantage(episode_traj, config):
    with torch.no_grad():
        lastgaelam = 0
        for t in reversed(range(config.training.rollout_steps)):
            if t == config.training.rollout_steps - 1:
                nextnonterminal = 0
                nextvalues = 0
            else:
                nextnonterminal = 1.0 - episode_traj["dones"][t + 1]
                # Use value over all options at next step as bootstrap target
                nextvalues = episode_traj["termination_values"][t + 1]  

            # # No explicit termination reward, so reward=0
            # delta = 0 + config.training.gamma * nextvalues * nextnonterminal - episode_traj["termination_values"][t]
            delta = episode_traj["termination_reward"][t] + config.training.gamma * nextvalues * nextnonterminal - episode_traj["termination_values"][t]

            episode_traj["termination_advantages"][t] = lastgaelam = (
                delta + config.training.gamma * config.training.gae_lambda * nextnonterminal * lastgaelam
            )
        episode_traj["termination_returns"] = episode_traj["termination_advantages"] + episode_traj["termination_values"]
    return episode_traj





def compute_advantage_term(episode_dict,hi_ep_dict, config):
    with torch.no_grad():
        lastgaelam = 0
        for t in reversed(range(config.training.rollout_steps)):
            if t == config.training.rollout_steps - 1:
                nextnonterminal = 0
                nextvalues = 0
            else:
                nextnonterminal = 1.0 - episode_dict["dones"][t + 1]
                nextvalues = episode_dict["termination_values"][t + 1]

            delta = (
                episode_dict["rewards"][t]
                + config.training.gamma * nextvalues * nextnonterminal
                - episode_dict["termination_values"][t]
            )
            episode_dict["termination_advantages"][t] = lastgaelam = (
                delta + config.training.gamma * config.training.gae_lambda * nextnonterminal * lastgaelam
            )

        episode_dict["termination_returns"] = (
            episode_dict["termination_advantages"] + episode_dict["termination_values"]
        )

    return episode_dict
def compute_termination_reward(step, episode_dict, term_action, delta_thresh=0.01, window_size=10):
    """
    Compute termination reward per env:
    +1 if:
      - term_action = 0 and reward improving → keep going
      - term_action = 1 and reward declining → switch
    -1 if:
      - term_action = 0 and reward declining → bad to continue
      - term_action = 1 and reward improving → bad to switch
    0 otherwise
    """
    rewards = episode_dict["rewards"]  # shape: [rollout_steps, num_envs]
    num_envs = rewards.shape[1]
    term_reward = torch.zeros(num_envs, device=rewards.device)

    for env in range(num_envs):
        # Define window boundaries
        curr_start = max(0, step - window_size + 1)
        prev_start = max(0, step - 2 * window_size + 1)
        prev_end   = max(0, step - window_size)

        # Extract reward windows
        current_window = rewards[curr_start:step+1, env]
        previous_window = rewards[prev_start:prev_end, env] if prev_end > prev_start else None

        if current_window.numel() == 0 or previous_window is None or previous_window.numel() == 0:
            continue  # Not enough data yet

        avg_current = current_window.mean()
        avg_previous = previous_window.mean()
        delta_r = avg_current - avg_previous

        if term_action[env] == 0:
            # Policy decided to continue
            if delta_r > delta_thresh:
                term_reward[env] = +1.0  # reward for staying on a good path
            elif delta_r < -delta_thresh:
                term_reward[env] = -1.0  # penalize continuing when reward is dropping
        elif term_action[env] == 1:
            # Policy decided to switch
            if delta_r < -delta_thresh:
                term_reward[env] = +1.0  # reward for switching when reward is dropping
            elif delta_r > delta_thresh:
                term_reward[env] = -1.0  # penalize switching when reward is increasing

    return term_reward  # shape: [num_envs]

def make_multi_view(features, labels, n_views=2):
    """
    Args:
        features: [N, d] tensor of embeddings
        labels: [N] tensor of skill labels
        n_views: number of views per sample (here 2)
    Returns:
        features_multi: [N, n_views, d]
    """
    N, d = features.shape
    device = features.device

    if n_views != 2:
        raise NotImplementedError("Only n_views=2 supported here")

    # Initialize multi-view tensor
    features_multi = torch.zeros((N, n_views, d), device=device)
    features_multi[:, 0, :] = features  # first view = original

    # Prepare second view: sample from same skill, different index
    for skill in torch.unique(labels):
        idx = (labels == skill).nonzero(as_tuple=True)[0]
        if len(idx) == 1:
            # Only one sample for this skill: duplicate itself
            features_multi[idx, 1, :] = features[idx]
        else:
            # Shuffle indices for pairing
            shuffled_idx = idx[torch.randperm(len(idx))]
            # Make sure no sample is paired with itself
            shuffled_idx = torch.roll(shuffled_idx, shifts=-1)
            features_multi[idx, 1, :] = features[shuffled_idx]

    return features_multi



class SupConReward(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConReward, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """
        Compute intrinsic rewards per anchor, instead of a scalar loss.

        Args:
            features: [bsz, n_views, d]
            labels: [bsz] (optional)
            mask: [bsz, bsz] (optional)
        Returns:
            rewards: [bsz] intrinsic reward for each anchor
        """
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown contrast_mode")

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute probabilities
        exp_logits = torch.exp(logits) * logits_mask
        probs = exp_logits / exp_logits.sum(1, keepdim=True)  # [N, N]

        # Reward = probability assigned to positives
        pos_probs = (mask * probs).sum(1) / mask.sum(1).clamp(min=1.0)

        rewards = pos_probs.view(anchor_count, batch_size).mean(0)  # [bsz]
        return rewards



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
