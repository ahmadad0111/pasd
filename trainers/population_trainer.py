import gym
import torch
import torch.nn as nn
import numpy as np
import os
import time

from utils.logger import Logger
from trainers.ppo import compute_ppo_loss
from tqdm import tqdm
from hydra.utils import instantiate, get_original_cwd
from torch.distributions.categorical import Categorical


class SPPopulationTrainer:
    def __init__(self, config, **kwargs):
        self.config = config
        self.device =  torch.device(f"cuda:{self.config.device_id}" if torch.cuda.is_available() and config.cuda else "cpu")
        self.agent_pop = [instantiate(config.model.agent, self.config).to(self.device) for agent in range(self.config.training.population_size)]
        self.optimizers = [instantiate(config.training.optimizer, self.agent_pop[idx].parameters(),lr=config.layout.lr, eps=1e-5) for idx,_ in enumerate(self.agent_pop)]
        self.runner = instantiate(config.model.runner, self.config, self.device)
        self.config.training.batch_size = int(2*self.config.training.rollout_steps * self.config.training.num_envs)
        self.config.training.minibatch_size = int(self.config.training.batch_size // self.config.training.num_minibatches)
        self.logger = Logger(self.config)
        self.global_step = 0
        self.best_average_rewards = [-np.inf for _ in range(self.config.training.population_size)]
    
    def run_episode(self, agent_pop, fracs):
        return self.runner.generate_episode(agent_pop, fracs)

    def prepare_batch(self, batch_trajs):
        return {
            "obs" : torch.cat((batch_trajs["p0"]["obs"].reshape((self.config.training.population_size, -1,) + tuple(self.config.layout.observation_shape)), batch_trajs["p1"]["obs"].reshape((self.config.training.population_size,-1,) + tuple(self.config.layout.observation_shape))),1),
            "logprobs" : torch.cat((batch_trajs["p0"]["logprobs"].reshape(self.config.training.population_size,-1),batch_trajs["p1"]["logprobs"].reshape(self.config.training.population_size,-1)),1),
            "actions" : torch.cat((batch_trajs["p0"]["actions"].reshape(self.config.training.population_size,-1,),batch_trajs["p1"]["actions"].reshape(self.config.training.population_size,-1)),1),
            "advantages" : torch.cat((batch_trajs["p0"]["advantages"].reshape(self.config.training.population_size,-1),batch_trajs["p1"]["advantages"].reshape(self.config.training.population_size,-1)),1),
            "returns" : torch.cat((batch_trajs["p0"]["returns"].reshape(self.config.training.population_size,-1),batch_trajs["p1"]["returns"].reshape(self.config.training.population_size,-1)),1),
            "values" : torch.cat((batch_trajs["p0"]["values"].reshape(self.config.training.population_size,-1),batch_trajs["p1"]["values"].reshape(self.config.training.population_size,-1)),1),
        }

    def train(self):
        last_save = self.config.logging.save_interval
        start_time = time.time()
        num_updates = int(2*self.config.layout.sp_timesteps // self.config.training.batch_size)
        for update in tqdm(range(1, num_updates + 1)):
            frac = 1.0 - (update - 1.0) / num_updates
            if self.config.training.anneal_lr:
                lrnow = self.config.layout.lr  - (1.0 - frac) * (self.config.layout.lr  - (self.config.layout.lr /self.config.layout.anneal_lr_fraction))
                for optimizer in self.optimizers:
                    optimizer.param_groups[0]["lr"] = lrnow

            if self.global_step < self.config.layout.rshaped_horizon:
                sr_frac = 1.0 - self.global_step/self.config.layout.rshaped_horizon
            else:
                sr_frac = 0

            trajs, infos_list = self.run_episode(self.agent_pop, (frac, sr_frac))

            self.global_step +=  self.config.training.num_envs * self.config.training.rollout_steps

            batch = self.prepare_batch(trajs)
            b_inds = np.arange(self.config.training.batch_size)
            pg_losses = []
            v_losses = []
            entropy_losses = []

            for epoch in range(self.config.training.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.config.training.batch_size, self.config.training.minibatch_size):
                    end = start + self.config.training.minibatch_size
                    mb_inds = b_inds[start:end]

                    loss = 0
                    avg_policy = torch.zeros((self.config.training.minibatch_size, self.config.layout.action_dim)).to(self.device)
                    avg_entropy = torch.zeros((self.config.training.minibatch_size)).to(self.device)

                    for agent_ind, agent in enumerate(self.agent_pop):
                        _, newlogprob, entropy, newvalue, policy = agent.get_action_and_value(
                            batch["obs"][agent_ind,mb_inds],
                            action = batch["actions"].long()[agent_ind, mb_inds],
                        )

                        pg_loss,v_loss, approx_kl, clipfracs = compute_ppo_loss(
                            newlogprob, 
                            batch["logprobs"][agent_ind, mb_inds],
                            batch["advantages"][agent_ind,mb_inds],
                            newvalue,
                            batch["values"][agent_ind, mb_inds],
                            batch["returns"][agent_ind, mb_inds],
                            self.config
                        )
                        entropy_loss = entropy.mean()

                        agent_loss = pg_loss - self.config.training.ent_coef_lo * entropy_loss + v_loss * self.config.training.value_coef
                        loss += agent_loss
                        avg_entropy += entropy
                        avg_policy += policy

                        self.optimizers[agent_ind].zero_grad()

                        if epoch == self.config.training.update_epochs - 1:
                            pg_losses.append(pg_loss.item())
                            v_losses.append(v_loss.item())
                            entropy_losses.append(entropy_loss.item())

                    avg_entropy = avg_entropy / self.config.training.population_size
                    avg_policy = avg_policy / self.config.training.population_size
                    
                    avg_policy = Categorical(probs=avg_policy)
                    entropy_of_avg = avg_policy.entropy()

                    JSD = (entropy_of_avg - avg_entropy).mean()

                    loss -= self.config.training.jsd_coef *JSD
                    loss.backward()

                    for agent_ind, agent in enumerate(self.agent_pop):
                        nn.utils.clip_grad_norm_(agent.parameters(), self.config.training.max_grad_norm)
                        self.optimizers[agent_ind].step()

                if self.config.training.target_kl is not None:
                    if approx_kl > self.config.training.target_kl:
                        break

            log_dict = {
                "trajs" : trajs,
                "infos" : infos_list,
                "pg_losses" : pg_losses,
                "v_losses" : v_losses,
                "entropy_losses" : entropy_losses,
                "entropy_of avg": entropy_of_avg.mean().item(),
                "avg_entropy" : avg_entropy.mean().item(),
                "jsd" : JSD.item(),
                "lr" : lrnow,
                "global_step" : self.global_step,
                "start_time" : start_time,
            }
            self.logger.log_train_pop_info(log_dict)
            
            for agent_ind, agent in enumerate(self.agent_pop):
                average_reward = np.mean([info["episode"]["r"] for info in infos_list[agent_ind]])
                if average_reward > self.best_average_rewards[agent_ind]:
                    self.best_average_rewards[agent_ind] = average_reward
                    self.save(f"agent{agent_ind}_best", os.getcwd(),agent_ind,save_model_state = True)

            if self.global_step >= last_save:
                last_save += self.config.logging.save_interval
                for agent_ind, agent in enumerate(self.agent_pop):
                    self.save(f"agent{agent_ind}_step{self.global_step}", os.getcwd(), agent_ind, save_model_state = True)
        
        for agent_ind, agent in enumerate(self.agent_pop):
            self.save(f"agent{agent_ind}_final", os.getcwd(), agent_ind, save_model_state = True)
        self.runner.envs.close()
        self.logger.writer.close()

    def save(self, agent_name, path, agent_ind, save_model_state = False):
        if save_model_state:
            torch.save({
                'timesteps': self.global_step,
                'model_state_dict': self.agent_pop[agent_ind].state_dict(),
                'optimizer_state_dict': self.optimizers[agent_ind].state_dict(),
                }, path + "/" + agent_name + "_model.pt"
            )
        else:
            torch.save(self.agent_pop[agent_ind].state_dict(), path + "/" + agent_name + ".pt")

    def load(self, agent_name, path, agent_ind, load_model_state=False):
        if load_model_state:
            checkpoint = torch.load(path + f"/agent{agent_ind}_{agent_name}" + "_model.pt")
            self.agent_pop[agent_ind].load_state_dict(checkpoint['model_state_dict'])
            self.optimizers[agent_ind].load_state_dict(checkpoint['optimizer_state_dict'])
            self.global_step = checkpoint['timesteps']
        else:
            self.agent_pop[agent_ind].load_state_dict(torch.load(path +  f"/agent{agent_ind}_{agent_name}" + "_model.pt"))

    def load_population(self, agent_name, path, load_model_state=True):
        for agent_ind, agent in enumerate(self.agent_pop):
            self.load(agent_name, get_original_cwd() + "/" + path, agent_ind, load_model_state=load_model_state)
            #self.load(agent_name,  path, agent_ind, load_model_state=load_model_state)

    
