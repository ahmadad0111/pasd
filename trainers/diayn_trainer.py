import itertools
import gym
import torch
import torch.nn as nn
import numpy as np
import os
import time

from utils.utils import get_partners
from utils.logger import Logger
from trainers.ppo import compute_ppo_loss
from tqdm import tqdm
from hydra.utils import instantiate

import torch.nn.functional as F



class DIAYNTrainer:
    def __init__(self, config, **kwargs):
        self.config = config
        self.device =  torch.device(f"cuda:{self.config.device_id}" if torch.cuda.is_available() and config.cuda else "cpu")
        self.hi_agent = instantiate(config.model.hi_agent, self.config).to(self.device)
        self.hi_optimizer = instantiate(config.training.optimizer, self.hi_agent.parameters(),lr=config.layout.lr, eps=1e-5)

        self.lo_agent = instantiate(config.model.lo_agent, self.config).to(self.device)
        self.lo_optimizer = instantiate(config.training.optimizer, self.lo_agent.parameters(),lr=config.layout.lr, eps=1e-5)

        self.discriminator = instantiate(config.model.discriminator, self.config).to(self.device)
        self.disc_optimizer = instantiate(config.training.optimizer, self.discriminator.parameters(),lr=config.layout.lr_disc, eps=1e-5)


        self.train_partners,self.test_partners = get_partners(config, self.device)
        self.runner = instantiate(config.model.runner, self.config, self.device)
        self.config.training.batch_size = int(self.config.training.rollout_steps * self.config.training.num_envs)
        self.config.training.minibatch_size = int(self.config.training.batch_size // self.config.training.num_minibatches)
        self.logger = Logger(self.config)
        self.global_step = 0
        self.best_average_reward = -np.inf
    
    def run_episode(self, lo_agent, hi_agent, discriminator, partners, lstm_state_lo, lstm_state_hi, lstm_state_disc, fracs,stage):
        #return self.runner.generate_episode(agents, partners, lstm_state, fracs)
        return self.runner.generate_episode(lo_agent, hi_agent, discriminator, partners, lstm_state_lo, lstm_state_hi, lstm_state_disc, fracs,stage)

    def prepare_batch(self, batch_trajs):
        batch_trajs["obs"] = torch.transpose(batch_trajs["obs"],0,1).reshape((-1,) + tuple(self.config.layout.observation_shape))
        batch_trajs["logprobs"] = torch.transpose(batch_trajs["logprobs"],0,1).reshape(-1)
        batch_trajs["actions"] = torch.transpose(batch_trajs["actions"],0,1).reshape((-1,))
        batch_trajs["current_zs"] = torch.transpose(batch_trajs["current_zs"],0,1).reshape((-1,))
        batch_trajs["dones"] = torch.transpose(batch_trajs["dones"],0,1).reshape(-1)
        batch_trajs["advantages"] = torch.transpose(batch_trajs["advantages"],0,1).reshape(-1)
        batch_trajs["returns"] = torch.transpose(batch_trajs["returns"],0,1).reshape(-1)
        batch_trajs["values"] = torch.transpose(batch_trajs["values"],0,1).reshape(-1)
        return batch_trajs


    # in stage 1 need to train lowlevel policy on intrinsic reward from discriminator . and also need to train discriminator
    def train_s1(self):
        start_time = time.time()
        num_updates = int(self.config.model.total_timesteps // self.config.training.batch_size)
        for update in tqdm(range(1, num_updates + 1)):
            frac = 1.0 - (update - 1.0) / num_updates
            if self.config.training.anneal_lr:
                lrnow = self.config.layout.lr  - (1.0 - frac) * (self.config.layout.lr  - (self.config.layout.lr /self.config.layout.anneal_lr_fraction))
                self.lo_optimizer.param_groups[0]["lr"] = lrnow

            if self.global_step < self.config.layout.rshaped_horizon:
                sr_frac = 1.0 - self.global_step/self.config.layout.rshaped_horizon
            else:
                sr_frac = 0

            lstm_state_lo = (
                torch.zeros(self.lo_agent.lstm.num_layers, self.config.training.num_envs, self.lo_agent.lstm.hidden_size).to(self.device),
                torch.zeros(self.lo_agent.lstm.num_layers, self.config.training.num_envs, self.lo_agent.lstm.hidden_size).to(self.device),
            )  

            lstm_state_disc = (
                torch.zeros(self.discriminator.lstm.num_layers, self.config.training.num_envs, self.discriminator.lstm.hidden_size).to(self.device),
                torch.zeros(self.discriminator.lstm.num_layers, self.config.training.num_envs, self.discriminator.lstm.hidden_size).to(self.device),
            )  

            lstm_state_hi = (
                torch.zeros(self.hi_agent.lstm.num_layers, self.config.training.num_envs, self.hi_agent.lstm.hidden_size).to(self.device),
                torch.zeros(self.hi_agent.lstm.num_layers, self.config.training.num_envs, self.hi_agent.lstm.hidden_size).to(self.device),
            )  

            lo_trajs, hi_trajs, infos, cum_kl_rew = self.run_episode(self.lo_agent,self.hi_agent, self.discriminator, self.train_partners, lstm_state_lo, lstm_state_hi, lstm_state_disc, (frac, sr_frac), stage = "s1")



            self.global_step +=  self.config.training.num_envs * self.config.training.rollout_steps

            lo_traj = self.prepare_batch(lo_trajs)
            envsperbatch =  self.config.training.num_envs // self.config.training.num_minibatches
            envinds = np.arange(self.config.training.num_envs)
            envinds_hi = np.arange(self.config.training.num_envs)
            flatinds = np.arange(self.config.training.batch_size).reshape(self.config.training.num_envs, self.config.training.rollout_steps)


            # training disc

            for epoch in range(self.config.training.disc_epochs):
                np.random.shuffle(envinds)
                total_correct = 0
                total_samples = 0
                for start in range(0,self.config.training.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[mbenvinds, :].ravel()


                    # Forward pass through discriminator
                    logits, _ = self.discriminator.get_disc_logits(
                        lo_traj["obs"][mb_inds],
                        lo_traj["dones"][mb_inds],
                        (lstm_state_disc[0][:, mbenvinds], lstm_state_disc[1][:, mbenvinds]),
                        env_first=True
                    )

                    # Cross-entropy loss
                    disc_loss = F.cross_entropy(logits, lo_traj["current_zs"][mb_inds].long())
                    
                    # accuracy
                    preds = logits.argmax(dim=-1)
                    batch_correct = (preds == lo_traj["current_zs"][mb_inds]).sum().item()

                    total_correct += batch_correct
                    total_samples += len(mb_inds)

                    # Backprop
                    self.disc_optimizer.zero_grad()
                    disc_loss.backward()
                    #nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
                    self.disc_optimizer.step()

                epoch_accuracy = total_correct / total_samples
                #print(f"Discriminator accuracy (epoch): {epoch_accuracy:.3f}")




            # train low policy
            for epoch in range(self.config.training.update_epochs):
                np.random.shuffle(envinds)
                np.random.shuffle(envinds_hi)
                for start in range(0, self.config.training.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[mbenvinds,:].ravel()


                    _, newlogprob, entropy, newvalue, __, ___,_ = self.lo_agent.get_action_and_value(
                        lo_traj["obs"][mb_inds],
                        lo_traj["current_zs"][mb_inds].long().reshape(-1),
                        lo_traj["dones"][mb_inds],
                        (lstm_state_lo[0][:, mbenvinds], lstm_state_lo[1][:, mbenvinds]),
                        action = lo_traj["actions"].long()[mb_inds],
                        env_first = True
                    )


                    lo_pg_loss,lo_v_loss, approx_kl, clipfracs = compute_ppo_loss(
                        newlogprob, 
                        lo_traj["logprobs"][mb_inds],
                        lo_traj["advantages"][mb_inds],
                        newvalue,
                        lo_traj["values"][mb_inds],
                        lo_traj["returns"][mb_inds],
                        self.config
                    )


                    lo_entropy_loss = entropy.mean()                      

                    loss_lo = lo_pg_loss - self.config.training.ent_coef_lo_s1 * lo_entropy_loss + lo_v_loss * self.config.training.value_coef

                    self.lo_optimizer.zero_grad()
                    loss_lo.backward()

                    nn.utils.clip_grad_norm_(self.lo_agent.parameters(), self.config.training.max_grad_norm)
                    self.lo_optimizer.step()

                # print(type(self.config.training.target_kl))
                if self.config.training.target_kl is not None:
                    if approx_kl > self.config.training.target_kl:
                        break

            y_pred, y_true = lo_traj["values"].cpu().numpy(), lo_traj["returns"].cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


            log_dict = {
                #"hi_traj" : hi_trajs,
                "lo_traj" : lo_trajs,
                "infos" : infos,
                "cum_kl_rew" : cum_kl_rew,
                "approx_kl" : approx_kl.item(),
                #"approx_kl_z" : approx_kl_z.item(),
                "clipfracs" : clipfracs,
                #"clipfracs_z" : clipfracs_z,
                "lo_pg_loss" : lo_pg_loss.item(),
                #"hi_pg_loss" : hi_pg_loss.item(),
                "lo_v_loss" : lo_v_loss.item(),
                #"hi_v_loss" : hi_v_loss.item(),
                "lo_entropy_loss" : lo_entropy_loss.item(),
                #"hi_entropy_loss" : hi_entropy_loss.item(),
                "explained_var" : explained_var,
                #"explained_var_z" : explained_var_z,
                "lr" : lrnow,
                "global_step" : self.global_step,
                "start_time" : start_time,
                "disc_acc" : epoch_accuracy,
            }
            self.logger.log_train_info_s1(log_dict)
                
            average_reward = np.mean([info["episode"]["r"] for info in infos])
            if average_reward > self.best_average_reward:
                self.best_average_reward = average_reward
                self.save("best_agent", os.getcwd(),save_model_state = True)
        
        self.save("final_agent", os.getcwd(), save_model_state = True)
        if not self.config.train_s2:
            self.runner.envs.close()
            self.logger.writer.close()

    # in stage 2 need to train high level metacontroller
    def train_s2(self):
        start_time = time.time()
        num_updates = int(self.config.model.total_timesteps // self.config.training.batch_size)
        for update in tqdm(range(1, num_updates + 1)):
            frac = 1.0 - (update - 1.0) / num_updates
            if self.config.training.anneal_lr:
                lrnow = self.config.layout.lr  - (1.0 - frac) * (self.config.layout.lr  - (self.config.layout.lr /self.config.layout.anneal_lr_fraction))
                self.lo_optimizer.param_groups[0]["lr"] = lrnow

            if self.global_step < self.config.layout.rshaped_horizon:
                sr_frac = 1.0 - self.global_step/self.config.layout.rshaped_horizon
            else:
                sr_frac = 0

            lstm_state_lo = (
                torch.zeros(self.lo_agent.lstm.num_layers, self.config.training.num_envs, self.lo_agent.lstm.hidden_size).to(self.device),
                torch.zeros(self.lo_agent.lstm.num_layers, self.config.training.num_envs, self.lo_agent.lstm.hidden_size).to(self.device),
            )  

            lstm_state_disc = (
                torch.zeros(self.discriminator.lstm.num_layers, self.config.training.num_envs, self.discriminator.lstm.hidden_size).to(self.device),
                torch.zeros(self.discriminator.lstm.num_layers, self.config.training.num_envs, self.discriminator.lstm.hidden_size).to(self.device),
            )  

            lstm_state_hi = (
                torch.zeros(self.hi_agent.lstm.num_layers, self.config.training.num_envs, self.hi_agent.lstm.hidden_size).to(self.device),
                torch.zeros(self.hi_agent.lstm.num_layers, self.config.training.num_envs, self.hi_agent.lstm.hidden_size).to(self.device),
            )  

            lo_trajs, hi_trajs, infos, cum_kl_rew = self.run_episode(self.lo_agent,self.hi_agent, self.discriminator, self.train_partners, lstm_state_lo, lstm_state_hi, lstm_state_disc, (frac, sr_frac), stage = "s2")



            self.global_step +=  self.config.training.num_envs * self.config.training.rollout_steps

            lo_traj = self.prepare_batch(lo_trajs)
            envsperbatch =  self.config.training.num_envs // self.config.training.num_minibatches
            envinds = np.arange(self.config.training.num_envs)
            envinds_hi = np.arange(self.config.training.num_envs)
            flatinds = np.arange(self.config.training.batch_size).reshape(self.config.training.num_envs, self.config.training.rollout_steps)


            # train meta policy

            for epoch in range(self.config.training.update_epochs):
                np.random.shuffle(envinds)
                np.random.shuffle(envinds_hi)
                for start in range(0, self.config.training.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbenvinds_hi = envinds_hi[start:end]
                    mb_inds = flatinds[mbenvinds,:].ravel()
                    mb_inds_hitolo = flatinds[mbenvinds_hi,:].ravel()

                    hi_mb_inds = list(itertools.chain.from_iterable([[step*len(mbenvinds_hi) + idx for step in hi_trajs[env]["inds"]] for idx, env in enumerate(mbenvinds_hi)]))

                    mb_z_s = torch.cat([hi_trajs[ind]["z_s"] for ind in mbenvinds_hi],-1)
                    mb_logprobs_z = torch.cat([hi_trajs[ind]["z_logprobs"] for ind in mbenvinds_hi],-1)
                    mb_advantages_z = torch.cat([hi_trajs[ind]["z_adv"] for ind in mbenvinds_hi],-1)
                    mb_returns_z = torch.cat([hi_trajs[ind]["z_ret"] for ind in mbenvinds_hi],-1)
                    mb_values_z = torch.cat([hi_trajs[ind]["z_values"] for ind in mbenvinds_hi],-1)

                    # _, newlogprob, entropy, newvalue, __, ___,_ = self.agent.get_action_and_value(
                    #     lo_traj["obs"][mb_inds],
                    #     lo_traj["current_zs"][mb_inds].long().reshape(-1),
                    #     lo_traj["dones"][mb_inds],
                    #     (lstm_state_lo[0][:, mbenvinds], lstm_state_lo[1][:, mbenvinds]),
                    #     action = lo_traj["actions"].long()[mb_inds],
                    #     env_first = True
                    # )

                    _, newlogprob_z, entropy_z, newvalue_z, __, ___ = self.hi_agent.get_z_and_value(
                        lo_traj["obs"][mb_inds_hitolo],
                        lo_traj["dones"][mb_inds_hitolo],
                        (lstm_state_hi[0][:, mbenvinds_hi], lstm_state_hi[1][:, mbenvinds_hi]),
                        z = mb_z_s.long(),
                        t_ind = hi_mb_inds,
                        env_first = True
                    )

                    # lo_pg_loss,lo_v_loss, approx_kl, clipfracs = compute_ppo_loss(
                    #     newlogprob, 
                    #     lo_traj["logprobs"][mb_inds],
                    #     lo_traj["advantages"][mb_inds],
                    #     newvalue,
                    #     lo_traj["values"][mb_inds],
                    #     lo_traj["returns"][mb_inds],
                    #     self.config
                    # )

                    hi_pg_loss, hi_v_loss, approx_kl_z, clipfracs_z = compute_ppo_loss(
                        newlogprob_z, 
                        mb_logprobs_z,
                        mb_advantages_z,
                        newvalue_z,
                        mb_values_z,
                        mb_returns_z,
                        self.config
                    )

                    #lo_entropy_loss = entropy.mean()
                    hi_entropy_loss = entropy_z.mean()                        

                    #loss_lo = lo_pg_loss - self.config.training.ent_coef_lo * lo_entropy_loss + lo_v_loss * self.config.training.value_coef
                    loss_hi = hi_pg_loss - self.config.training.ent_coef_hi * hi_entropy_loss + hi_v_loss * self.config.training.value_coef

                    
                    self.hi_optimizer.zero_grad()
                    loss_hi.backward()

                    nn.utils.clip_grad_norm_(self.hi_agent.parameters(), self.config.training.max_grad_norm)
                    self.hi_optimizer.step()

                # print(type(self.config.training.target_kl))
                # if self.config.training.target_kl is not None:
                #     if approx_kl > self.config.training.target_kl:
                #         break

            y_pred, y_true = lo_traj["values"].cpu().numpy(), lo_traj["returns"].cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            b_returns_z = torch.cat([hi_trajs[ind]["z_ret"] for ind in range(self.config.training.num_envs)],-1)
            b_values_z = torch.cat([hi_trajs[ind]["z_values"] for ind in range(self.config.training.num_envs)],-1)

            y_pred_z, y_true_z = b_values_z.cpu().numpy(), b_returns_z.cpu().numpy()
            var_y_z = np.var(y_true_z)
            explained_var_z = np.nan if var_y_z == 0 else 1 - np.var(y_true_z - y_pred_z) / var_y_z

            log_dict = {
                "hi_traj" : hi_trajs,
                "lo_traj" : lo_trajs,
                "infos" : infos,
                "cum_kl_rew" : cum_kl_rew,
                #"approx_kl" : approx_kl.item(),
                "approx_kl_z" : approx_kl_z.item(),
                #"clipfracs" : clipfracs,
                "clipfracs_z" : clipfracs_z,
                #"lo_pg_loss" : lo_pg_loss.item(),
                "hi_pg_loss" : hi_pg_loss.item(),
                #"lo_v_loss" : lo_v_loss.item(),
                "hi_v_loss" : hi_v_loss.item(),
                #"lo_entropy_loss" : lo_entropy_loss.item(),
                "hi_entropy_loss" : hi_entropy_loss.item(),
                "explained_var" : explained_var,
                "explained_var_z" : explained_var_z,
                "lr" : lrnow,
                "global_step" : self.global_step,
                "start_time" : start_time,
            }
            self.logger.diayn_log_train_info_s2(log_dict)
                
            average_reward = np.mean([info["episode"]["r"] for info in infos])
            if average_reward > self.best_average_reward:
                self.best_average_reward = average_reward
                self.save("best_agent", os.getcwd(),save_model_state = True)
        
        self.save("final_agent", os.getcwd(), save_model_state = True)
        self.runner.envs.close()
        self.logger.writer.close()



    
    def train(self):
        start_time = time.time()
        num_updates = int(self.config.model.total_timesteps // self.config.training.batch_size)
        for update in tqdm(range(1, num_updates + 1)):
            frac = 1.0 - (update - 1.0) / num_updates
            if self.config.training.anneal_lr:
                lrnow = self.config.layout.lr  - (1.0 - frac) * (self.config.layout.lr  - (self.config.layout.lr /self.config.layout.anneal_lr_fraction))
                self.optimizer.param_groups[0]["lr"] = lrnow

            if self.global_step < self.config.layout.rshaped_horizon:
                sr_frac = 1.0 - self.global_step/self.config.layout.rshaped_horizon
            else:
                sr_frac = 0

            lstm_state = (
                torch.zeros(self.agent.lstm.num_layers, self.config.training.num_envs, self.agent.lstm.hidden_size).to(self.device),
                torch.zeros(self.agent.lstm.num_layers, self.config.training.num_envs, self.agent.lstm.hidden_size).to(self.device),
            )   

            lo_trajs, hi_trajs, infos, cum_kl_rew = self.run_episode(self.agent, self.train_partners, lstm_state, (frac, sr_frac))

            self.global_step +=  self.config.training.num_envs * self.config.training.rollout_steps

            lo_traj = self.prepare_batch(lo_trajs)
            envsperbatch =  self.config.training.num_envs // self.config.training.num_minibatches
            envinds = np.arange(self.config.training.num_envs)
            envinds_hi = np.arange(self.config.training.num_envs)
            flatinds = np.arange(self.config.training.batch_size).reshape(self.config.training.num_envs, self.config.training.rollout_steps)

            for epoch in range(self.config.training.update_epochs):
                np.random.shuffle(envinds)
                np.random.shuffle(envinds_hi)
                for start in range(0, self.config.training.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbenvinds_hi = envinds_hi[start:end]
                    mb_inds = flatinds[mbenvinds,:].ravel()
                    mb_inds_hitolo = flatinds[mbenvinds_hi,:].ravel()

                    hi_mb_inds = list(itertools.chain.from_iterable([[step*len(mbenvinds_hi) + idx for step in hi_trajs[env]["inds"]] for idx, env in enumerate(mbenvinds_hi)]))

                    mb_z_s = torch.cat([hi_trajs[ind]["z_s"] for ind in mbenvinds_hi],-1)
                    mb_logprobs_z = torch.cat([hi_trajs[ind]["z_logprobs"] for ind in mbenvinds_hi],-1)
                    mb_advantages_z = torch.cat([hi_trajs[ind]["z_adv"] for ind in mbenvinds_hi],-1)
                    mb_returns_z = torch.cat([hi_trajs[ind]["z_ret"] for ind in mbenvinds_hi],-1)
                    mb_values_z = torch.cat([hi_trajs[ind]["z_values"] for ind in mbenvinds_hi],-1)

                    _, newlogprob, entropy, newvalue, __, ___ = self.agent.get_action_and_value(
                        lo_traj["obs"][mb_inds],
                        lo_traj["current_zs"][mb_inds].long().reshape(-1),
                        lo_traj["dones"][mb_inds],
                        (lstm_state[0][:, mbenvinds], lstm_state[1][:, mbenvinds]),
                        action = lo_traj["actions"].long()[mb_inds],
                        env_first = True
                    )

                    _, newlogprob_z, entropy_z, newvalue_z, __, ___ = self.agent.get_z_and_value(
                        lo_traj["obs"][mb_inds_hitolo],
                        lo_traj["dones"][mb_inds_hitolo],
                        (lstm_state[0][:, mbenvinds_hi], lstm_state[1][:, mbenvinds_hi]),
                        z = mb_z_s.long(),
                        t_ind = hi_mb_inds,
                        env_first = True
                    )

                    lo_pg_loss,lo_v_loss, approx_kl, clipfracs = compute_ppo_loss(
                        newlogprob, 
                        lo_traj["logprobs"][mb_inds],
                        lo_traj["advantages"][mb_inds],
                        newvalue,
                        lo_traj["values"][mb_inds],
                        lo_traj["returns"][mb_inds],
                        self.config
                    )

                    hi_pg_loss, hi_v_loss, approx_kl_z, clipfracs_z = compute_ppo_loss(
                        newlogprob_z, 
                        mb_logprobs_z,
                        mb_advantages_z,
                        newvalue_z,
                        mb_values_z,
                        mb_returns_z,
                        self.config
                    )

                    lo_entropy_loss = entropy.mean()
                    hi_entropy_loss = entropy_z.mean()                        

                    loss_lo = lo_pg_loss - self.config.training.ent_coef_lo * lo_entropy_loss + lo_v_loss * self.config.training.value_coef
                    loss_hi = hi_pg_loss - self.config.training.ent_coef_hi * hi_entropy_loss + hi_v_loss * self.config.training.value_coef

                    hippo_loss = loss_lo + loss_hi
                    self.optimizer.zero_grad()
                    hippo_loss.backward()

                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.training.max_grad_norm)
                    self.optimizer.step()

                # print(type(self.config.training.target_kl))
                if self.config.training.target_kl is not None:
                    if approx_kl > self.config.training.target_kl:
                        break

            y_pred, y_true = lo_traj["values"].cpu().numpy(), lo_traj["returns"].cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            b_returns_z = torch.cat([hi_trajs[ind]["z_ret"] for ind in range(self.config.training.num_envs)],-1)
            b_values_z = torch.cat([hi_trajs[ind]["z_values"] for ind in range(self.config.training.num_envs)],-1)

            y_pred_z, y_true_z = b_values_z.cpu().numpy(), b_returns_z.cpu().numpy()
            var_y_z = np.var(y_true_z)
            explained_var_z = np.nan if var_y_z == 0 else 1 - np.var(y_true_z - y_pred_z) / var_y_z

            log_dict = {
                "hi_traj" : hi_trajs,
                "lo_traj" : lo_trajs,
                "infos" : infos,
                "cum_kl_rew" : cum_kl_rew,
                "approx_kl" : approx_kl.item(),
                "approx_kl_z" : approx_kl_z.item(),
                "clipfracs" : clipfracs,
                "clipfracs_z" : clipfracs_z,
                "lo_pg_loss" : lo_pg_loss.item(),
                "hi_pg_loss" : hi_pg_loss.item(),
                "lo_v_loss" : lo_v_loss.item(),
                "hi_v_loss" : hi_v_loss.item(),
                "lo_entropy_loss" : lo_entropy_loss.item(),
                "hi_entropy_loss" : hi_entropy_loss.item(),
                "explained_var" : explained_var,
                "explained_var_z" : explained_var_z,
                "lr" : lrnow,
                "global_step" : self.global_step,
                "start_time" : start_time,
            }
            self.logger.log_train_info_s1(log_dict)
                
            average_reward = np.mean([info["episode"]["r"] for info in infos])
            if average_reward > self.best_average_reward:
                self.best_average_reward = average_reward
                self.save("best_agent", os.getcwd(),save_model_state = True)
        
        self.save("final_agent", os.getcwd(), save_model_state = True)
        self.runner.envs.close()
        self.logger.writer.close()

    def save(self, agent_name, path, save_model_state = False):
        if save_model_state:
            torch.save({
                'timesteps': self.global_step,
                'lo_model_state_dict': self.lo_agent.state_dict(),
                'lo_optimizer_state_dict': self.lo_optimizer.state_dict(),
                'hi_model_state_dict': self.hi_agent.state_dict(),
                'hi_optimizer_state_dict': self.hi_optimizer.state_dict(),

                'disc_state_dict': self.discriminator.state_dict(),
                'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
                }, path + "/" + agent_name + "_model.pt"
            )
        else:
            torch.save(self.lo_agent.state_dict(), path + "/" + agent_name + "_lo.pt")
            torch.save(self.hi_agent.state_dict(), path + "/" + agent_name + "_hi.pt")
            torch.save(self.discriminator.state_dict(), path + "/" + agent_name + "_disc.pt")
            

    def load_model_state(self, agent_name, path):
        checkpoint = torch.load(path + "/" + agent_name + "_model.pt")
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['timesteps']

    def load_model_state_diayn(self, agent_name, path):
        checkpoint = torch.load(path + "/" + agent_name + "_model.pt")
        self.lo_agent.load_state_dict(checkpoint['lo_model_state_dict'])
        self.lo_optimizer.load_state_dict(checkpoint['lo_optimizer_state_dict'])

        self.hi_agent.load_state_dict(checkpoint['hi_model_state_dict'])
        self.hi_optimizer.load_state_dict(checkpoint['hi_optimizer_state_dict'])

        self.discriminator.load_state_dict(checkpoint['disc_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        self.global_step = checkpoint['timesteps']


    def load(self, agent_name, path):
        self.agent.load_state_dict(torch.load(path + "/" + agent_name + ".pt"))

    
