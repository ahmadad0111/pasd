import gym
import numpy as np
import torch
import torch.nn as nn
from runners.utils import compute_advantage
from utils.utils import generate_ep_partners, make_env

from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


class FCPRunner:
    def __init__(self, config, device, **kwargs):
        self.config = config
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(config.gym_id, config.layout.name, config.seed + i) for i in range(self.config.training.num_envs)]
        )
        self.initial_obs = self.envs.reset()
        self.device = device

    def make_episode_dict(self):
        return {
            "obs": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs) + tuple(self.config.layout.observation_shape)).to(self.device),
            "actions": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            "rewards": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            "values": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            "logprobs": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            "advantages": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            "returns": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            "dones": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
        }


    def generate_episode(self, agent, partner_agents,lstm_state, fracs):
        episode_dict = self.make_episode_dict()
        episode_dict, info = self.rollout(agent, partner_agents, episode_dict, lstm_state, fracs)
        episode_dict = compute_advantage(episode_dict,self.config)
        return episode_dict, info

    def rollout(self, agent, partner_agents, episode_dict, lstm_state, fracs):

        partner_idxs,partner_roles,player_roles = generate_ep_partners(partner_agents, self.config.training.num_envs)
        agent_roles = 1 - partner_roles
        
        partners, counts = np.unique(partner_idxs, return_counts = True)

        next_obs_p0 = torch.Tensor(self.initial_obs[:,0]).to(self.device)
        next_obs_p1 = torch.Tensor(self.initial_obs[:,1]).to(self.device)

        def get_agent_obs(obs_p0,obs_p1):
            obs_list = []
            obs_agent = torch.zeros((self.config.training.num_envs,) + tuple(self.config.layout.observation_shape)).to(self.device)
            for ind,partner in enumerate(partners):
                obs_list.append(torch.Tensor([]).to(self.device))
            for env in range(self.config.training.num_envs):
                pobs_idx = np.where(partners == partner_idxs[env])[0][0]
                if partner_roles[env] == 0:
                    obs_agent[env] = torch.squeeze(next_obs_p1[env:env+1])
                    obs_list[pobs_idx] = torch.cat((obs_list[pobs_idx], obs_p0[env:env+1]))
                else:
                    obs_agent[env] = torch.squeeze(next_obs_p0[env:env+1])
                    obs_list[pobs_idx] = torch.cat((obs_list[pobs_idx], obs_p1[env:env+1]))
            return obs_agent, obs_list

        obs_agent, obs_list = get_agent_obs(next_obs_p0,next_obs_p1)
        next_done = torch.zeros(self.config.training.num_envs).to(self.device)

        for step in range(self.config.training.rollout_steps):
            episode_dict["dones"][step] = next_done
            current_lstm_state = (lstm_state[0].clone(), lstm_state[1].clone())

            pacts_list = []

            with torch.no_grad():
                actions_agent, logprob_agent, _, values_agent, lstm_state = agent.get_action_and_value(obs_agent, next_done, lstm_state)
                for ind,partner in enumerate(partners):
                    partner_actions,__, _, ___, ____ = partner_agents[partner].get_action_and_value(obs_list[ind])
                    pacts_list.append(partner_actions)

            partner_counts = np.zeros(len(partners), dtype = np.int64)
            actions_p0 = torch.zeros((self.config.training.num_envs)).to(self.device)
            actions_p1 = torch.zeros((self.config.training.num_envs)).to(self.device)

            for env in range(self.config.training.num_envs):
                pobs_idx = np.where(partners == partner_idxs[env])[0][0]
                if partner_roles[env] == 0:
                    action_p0 = pacts_list[pobs_idx][partner_counts[pobs_idx]]
                    action_p1 = actions_agent[env]
                else:
                    action_p0 = actions_agent[env]
                    action_p1 = pacts_list[pobs_idx][partner_counts[pobs_idx]]
                partner_counts[pobs_idx] += 1
                actions_p0[env] = action_p0
                actions_p1[env] = action_p1

            episode_dict["obs"][step] = obs_agent
            episode_dict["actions"][step] = actions_agent
            episode_dict["logprobs"][step] = logprob_agent
            episode_dict["values"][step] = values_agent.flatten()

            joint_action = torch.cat((actions_p0.view(self.config.training.num_envs,1),actions_p1.view(self.config.training.num_envs,1)),1)
            joint_action = joint_action.type(torch.int8)
            next_obs, reward, done, info = self.envs.step(joint_action.cpu().numpy())

            next_obs_p0 = torch.Tensor(next_obs[:,0]).to(self.device)
            next_obs_p1 = torch.Tensor(next_obs[:,1]).to(self.device)
            next_done = torch.Tensor(done).to(self.device)

            obs_agent, obs_list = get_agent_obs(next_obs_p0,next_obs_p1)

            shaped_r = np.zeros(self.config.training.num_envs)
            role_r = np.zeros(self.config.training.num_envs)

            for ind,item in enumerate(info):
                if agent_roles[ind] == 0:
                    shaped_r[ind] = item["shaped_r_by_agent"][0]
                    role_r[ind] = item["role_r_by_agent"][0]
                else:
                    shaped_r[ind] = item["shaped_r_by_agent"][1]
                    role_r[ind] = item["role_r_by_agent"][1]

            agent_reward = np.zeros(self.config.training.num_envs)

            agent_reward = reward
            if self.config.layout.soup_reward:
                if self.config.layout.soup_reward_decay:
                    agent_reward += fracs[1] *role_r
                else:
                    agent_reward += role_r
            if self.config.layout.ingred_reward:
                if self.config.layout.ingred_reward_decay:
                    agent_reward += fracs[1] *shaped_r
                else:
                    agent_reward += shaped_r

            agent_reward = torch.tensor(agent_reward).to(self.device).view(-1)
            episode_dict["rewards"][step] = agent_reward

        self.initial_obs = next_obs
        

        return episode_dict, info