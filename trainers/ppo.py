import numpy as np
import torch
import torch.nn as nn

def compute_ppo_loss(newlogprob, mblogprob, mb_advantages, newvalue, mb_values, mb_returns, config):
    logratio = newlogprob - mblogprob
    ratio = logratio.exp()
    clipfracs = []

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfracs = [((ratio - 1.0).abs() > config.training.clip_coef).float().mean().item()]

    if config.training.norm_adv:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.training.clip_coef, 1 + config.training.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    newvalue = newvalue.view(-1)
    if config.training.clip_vloss:
        v_loss_unclipped = (newvalue - mb_returns) ** 2
        v_clipped =  mb_values + torch.clamp(
            newvalue - mb_values,
            -config.training.clip_coef,
            config.training.clip_coef,
        )

        v_loss_clipped = (v_clipped - mb_returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    return pg_loss, v_loss, approx_kl, clipfracs
