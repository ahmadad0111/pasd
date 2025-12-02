import gym
import numpy as np
import torch
import torch.nn as nn
from runners.utils import compute_advantage_pop
from utils.utils import make_env

class SPRunner:
    def __init__(self, config, device, **kwargs):
        self.config = config

        print(f"layouit name is : {config.layout.name}")
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(config.gym_id, config.layout.name, config.seed + i) for i in range(self.config.training.num_envs)]
        )
        self.initial_obs = self.envs.reset()
        self.device = device

    def make_episode_dict(self):
        return {
            "p0": {
                "obs" : torch.zeros((self.config.training.population_size, self.config.training.rollout_steps, self.config.training.num_envs) + tuple(self.config.layout.observation_shape)).to(self.device),
                "actions": torch.zeros((self.config.training.population_size,self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
                "values": torch.zeros((self.config.training.population_size, self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
                "logprobs": torch.zeros((self.config.training.population_size, self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
                "advantages": torch.zeros((self.config.training.population_size,self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
                "returns": torch.zeros((self.config.training.population_size, self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            },
            "p1": {
                "obs" : torch.zeros((self.config.training.population_size, self.config.training.rollout_steps, self.config.training.num_envs) + tuple(self.config.layout.observation_shape)).to(self.device),
                "actions": torch.zeros((self.config.training.population_size,self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
                "values": torch.zeros((self.config.training.population_size, self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
                "logprobs": torch.zeros((self.config.training.population_size, self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
                "advantages": torch.zeros((self.config.training.population_size,self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
                "returns": torch.zeros((self.config.training.population_size,self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            },
            "rewards": torch.zeros((self.config.training.population_size,self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            "dones": torch.zeros((self.config.training.population_size,self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
        }

    def generate_episode(self, agent_pop, fracs):
        episode_dict = self.make_episode_dict()
        episode_dict, info = self.rollout(agent_pop, episode_dict, fracs)
        episode_dict = compute_advantage_pop(episode_dict,self.config)
        return episode_dict, info

    def rollout(self, agent_pop, episode_dict, fracs):

        next_obs_p0 = torch.Tensor(self.initial_obs[:,0]).to(self.device)
        next_obs_p1 = torch.Tensor(self.initial_obs[:,1]).to(self.device)

        next_done = torch.zeros(self.config.training.num_envs).to(self.device)
        info_list = []

        for agent_ind, agent in enumerate(agent_pop):
            for step in range(self.config.training.rollout_steps):
                episode_dict["dones"][agent_ind, step] = next_done

                pacts_list = []
                episode_dict["p0"]["obs"][agent_ind, step] = next_obs_p0
                episode_dict["p1"]["obs"][agent_ind, step] = next_obs_p1

                with torch.no_grad():
                    actions, logprobs, _, values, __ = agent_pop[agent_ind].get_action_and_value(torch.cat((next_obs_p0, next_obs_p1), dim= 0))
                    episode_dict["p0"]["values"][agent_ind,step] = values[:self.config.training.num_envs].flatten()
                    episode_dict["p1"]["values"][agent_ind,step] = values[self.config.training.num_envs:].flatten()

                episode_dict["p0"]["actions"][agent_ind,step] = actions[:self.config.training.num_envs]
                episode_dict["p1"]["actions"][agent_ind,step] = actions[self.config.training.num_envs:]
                episode_dict["p0"]["logprobs"][agent_ind,step] = logprobs[:self.config.training.num_envs]
                episode_dict["p1"]["logprobs"][agent_ind,step] = logprobs[self.config.training.num_envs:]

                joint_action = torch.cat((actions[:self.config.training.num_envs].view(self.config.training.num_envs,1),actions[self.config.training.num_envs:].view(self.config.training.num_envs,1)),1)
                joint_action = joint_action.type(torch.int8)
                next_obs, reward, done, info = self.envs.step(joint_action.cpu().numpy())

                next_obs_p0 = torch.Tensor(next_obs[:,0]).to(self.device)
                next_obs_p1 = torch.Tensor(next_obs[:,1]).to(self.device)
                next_done = torch.Tensor(done).to(self.device)

                shaped_r = np.zeros(self.config.training.num_envs)
                role_r = np.zeros(self.config.training.num_envs)

                for ind,item in enumerate(info):
                    shaped_r[ind] = sum(item["shaped_r_by_agent"])
              
                agent_reward = np.zeros(self.config.training.num_envs)

                agent_reward = reward

                if self.config.layout.ingred_reward:
                    agent_reward += fracs[1] *shaped_r

                agent_reward = torch.tensor(agent_reward).to(self.device).view(-1)
                episode_dict["rewards"][agent_ind,step] = agent_reward

            info_list.append(info)

        self.initial_obs = next_obs
        
        return episode_dict, info_list