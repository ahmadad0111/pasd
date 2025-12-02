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


class FCPTrainer:
    def __init__(self, config, **kwargs):
        self.config = config
        self.device =  torch.device(f"cuda:{self.config.device_id}" if torch.cuda.is_available() and config.cuda else "cpu")
        self.agent = instantiate(config.model.agent, self.config).to(self.device)
        self.optimizer = instantiate(config.training.optimizer, self.agent.parameters(),lr=config.layout.lr, eps=1e-5)
        self.train_partners,self.test_partners = get_partners(config, self.device)
        self.runner = instantiate(config.model.runner, self.config, self.device)
        self.config.training.batch_size = int(self.config.training.rollout_steps * self.config.training.num_envs)
        self.config.training.minibatch_size = int(self.config.training.batch_size // self.config.training.num_minibatches)
        self.logger = Logger(self.config)
        self.global_step = 0
        self.best_average_reward = -np.inf
    
    def run_episode(self, agents, partners, lstm_state, fracs):
        return self.runner.generate_episode(agents, partners, lstm_state, fracs)

    def prepare_batch(self, batch_trajs):
        batch_trajs["obs"] = batch_trajs["obs"].reshape((-1,) + tuple(self.config.layout.observation_shape))
        batch_trajs["logprobs"] = batch_trajs["logprobs"].reshape(-1)
        batch_trajs["actions"] = batch_trajs["actions"].reshape((-1,))
        batch_trajs["dones"] = batch_trajs["dones"].reshape(-1)
        batch_trajs["advantages"] = batch_trajs["advantages"].reshape(-1)
        batch_trajs["returns"] = batch_trajs["returns"].reshape(-1)
        batch_trajs["values"] = batch_trajs["values"].reshape(-1)
        return batch_trajs

    
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

            trajs, infos = self.run_episode(self.agent, self.train_partners, lstm_state, (frac, sr_frac))

            self.global_step +=  self.config.training.num_envs * self.config.training.rollout_steps

            trajs = self.prepare_batch(trajs)
            envsperbatch =  self.config.training.num_envs // self.config.training.num_minibatches
            envinds = np.arange(self.config.training.num_envs)
            flatinds = np.arange(self.config.training.batch_size).reshape(self.config.training.rollout_steps, self.config.training.num_envs)
            b_inds = np.arange(self.config.training.batch_size)

            for epoch in range(self.config.training.update_epochs):
                np.random.shuffle(envinds)
                for start in range(0, self.config.training.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:,mbenvinds].ravel()

                    _, newlogprob, entropy, newvalue, __ = self.agent.get_action_and_value(
                        trajs["obs"][mb_inds],
                        trajs["dones"][mb_inds],
                        (lstm_state[0][:, mbenvinds], lstm_state[1][:, mbenvinds]),
                        action = trajs["actions"].long()[mb_inds],
                    )

                    pg_loss,v_loss, approx_kl, clipfracs = compute_ppo_loss(
                        newlogprob, 
                        trajs["logprobs"][mb_inds],
                        trajs["advantages"][mb_inds],
                        newvalue,
                        trajs["values"][mb_inds],
                        trajs["returns"][mb_inds],
                        self.config
                    )

                    entropy_loss = entropy.mean()

                    loss = pg_loss - self.config.training.ent_coef_lo * entropy_loss + v_loss * self.config.training.value_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.training.max_grad_norm)
                    self.optimizer.step()

                # print(type(self.config.training.target_kl))
                if self.config.training.target_kl is not None:
                    if approx_kl > self.config.training.target_kl:
                        break

            log_dict = {
                "trajs" : trajs,
                "infos" : infos,
                "approx_kl" : approx_kl.item(),
                "clipfracs" : clipfracs,
                "pg_loss" : pg_loss.item(),
                "v_loss" : v_loss.item(),
                "entropy_loss" : entropy_loss.item(),
                "lr" : lrnow,
                "global_step" : self.global_step,
                "start_time" : start_time,
            }
            self.logger.log_train_info(log_dict)
                
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
                'model_state_dict': self.agent.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, path + "/" + agent_name + "_model.pt"
            )
        else:
            torch.save(self.agent.state_dict(), path + "/" + agent_name + ".pt")

    def load_model_state(self, agent_name, path):
        checkpoint = torch.load(path + "/" + agent_name + "_model.pt")
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['timesteps']

    def load(self, agent_name, path):
        self.agent.load_state_dict(torch.load(path + "/" + agent_name + ".pt"))
