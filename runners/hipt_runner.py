import gym
import numpy as np
import torch
import torch.nn as nn
from runners.utils import compute_advantage, compute_advantage_hi
from utils.utils import generate_ep_partners, make_env

class HiPTRunner:
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
            "current_zs": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
        }

    def make_episode_hi_dict(self):
        hi_ep = {}
        for env in range(self.config.training.num_envs):
            hi_ep[env] = {}
            hi_ep[env]["inds"] = []
        return hi_ep

    def generate_episode(self, agent, partner_agents,lstm_state, fracs):
        episode_dict = self.make_episode_dict()
        hi_ep_dict = self.make_episode_hi_dict()
        episode_dict, hi_ep_dict, info, cum_kl_rew = self.rollout(agent, partner_agents, episode_dict, hi_ep_dict, lstm_state, fracs)
        for ind in range(self.config.training.num_envs):
            hi_ep_dict[ind]["z_adv"] = torch.zeros_like(hi_ep_dict[ind]["hi_rewards"]).to(self.device)
        episode_dict = compute_advantage(episode_dict,self.config)
        hi_ep_dict = compute_advantage_hi(hi_ep_dict,self.config)
        return episode_dict, hi_ep_dict, info, cum_kl_rew

    def rollout(self, agent, partner_agents, episode_dict, hi_ep_dict, lstm_state, fracs):

        partner_idxs,partner_roles,player_roles = generate_ep_partners(partner_agents, self.config.training.num_envs)
        agent_roles = 1 - partner_roles
        next_terminations = np.zeros(self.config.training.num_envs)
        switch_ind = np.zeros(self.config.training.num_envs)
        curr_p_len = np.zeros(self.config.training.num_envs)
        hi_cum_rew = torch.zeros((self.config.training.num_envs)).to(self.device)
        current_z = torch.zeros((self.config.training.num_envs), dtype = torch.int64).to(self.device)
        expanded_z = torch.tile(torch.arange(self.config.layout.z_dim).view(-1,1),(self.config.training.num_envs,)).reshape(-1).to(self.device)
        cum_kl_rew = np.zeros(self.config.training.num_envs)

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

            with torch.no_grad():
                z, logprob_z, _, value_z, lstm_state, z_prob = agent.get_z_and_value(obs_agent, next_done, current_lstm_state)

            if step == 0:
                new_p_steps = np.random.randint(self.config.training.p_range[0], self.config.training.p_range[1], size = self.config.training.num_envs)
                next_terminations = new_p_steps
                curr_p_len = new_p_steps
                current_z = z.clone()

                for ind in range(self.config.training.num_envs):
                    hi_ep_dict[ind]["obs"] = obs_agent[ind].view((1,) + tuple(self.config.layout.observation_shape))
                    hi_ep_dict[ind]["z_s"] = z[ind].view(1)
                    hi_ep_dict[ind]["z_logprobs"] = logprob_z[ind].view(1)
                    hi_ep_dict[ind]["z_values"] = value_z[ind]
                    hi_ep_dict[ind]["inds"].append(step)
            else:
                for ind, pstep in enumerate(next_terminations):
                    if pstep == 0:
                        new_p_step = np.random.randint(self.config.training.p_range[0], self.config.training.p_range[1])
                        next_terminations[ind] = new_p_step
                        current_z[ind] = z[ind]
                        switch_ind[ind] += 1
                        hi_ep_dict[ind]["obs"] = torch.cat((hi_ep_dict[ind]["obs"],obs_agent[ind].view((1,) + tuple(self.config.layout.observation_shape))),0)
                        hi_ep_dict[ind]["z_s"] = torch.cat((hi_ep_dict[ind]["z_s"],z[ind].view(-1)),-1)
                        hi_ep_dict[ind]["z_logprobs"] = torch.cat((hi_ep_dict[ind]["z_logprobs"],logprob_z[ind].view(-1)),-1)
                        hi_ep_dict[ind]["z_values"] = torch.cat((hi_ep_dict[ind]["z_values"],value_z[ind]),-1)
                        hi_ep_dict[ind]["inds"].append(step)

            episode_dict["current_zs"][step] = current_z

            pacts_list = []

            with torch.no_grad():
                actions_agent_full, logprob_agent_full, _, value_agent_full, __, a_prob_full = agent.get_action_and_value(obs_agent, expanded_z, next_done, current_lstm_state, expanded = True)
                for ind,partner in enumerate(partners):
                    partner_actions,__, _, ___, ____ = partner_agents[partner].get_action_and_value(obs_list[ind])
                    pacts_list.append(partner_actions)

            actions_agent_full = actions_agent_full.reshape(self.config.layout.z_dim, self.config.training.num_envs)
            logprob_agent_full = logprob_agent_full.reshape(self.config.layout.z_dim, self.config.training.num_envs)
            value_agent_full = value_agent_full.flatten().reshape(self.config.layout.z_dim, self.config.training.num_envs)
            a_prob_full = a_prob_full.reshape(self.config.layout.z_dim, self.config.training.num_envs, -1)

            actions_agent = torch.gather(actions_agent_full, 0,current_z.view(1,-1)).reshape(-1)
            logprob_agent = torch.gather(logprob_agent_full, 0,current_z.view(1,-1)).reshape(-1)
            value_agent = torch.gather(value_agent_full, 0,current_z.view(1,-1)).reshape(-1)
            a_prob = torch.gather(a_prob_full, 0, torch.tile(current_z.view(1,-1,1),(1,1,self.config.layout.action_dim))).squeeze()

            a_prob_full = torch.cat([a_prob_full[i].view(self.config.training.num_envs,1,-1) for i in range(self.config.layout.z_dim,)],1)
            z_prob = z_prob.view(self.config.training.num_envs,1,-1)
            a_marginal = torch.matmul(z_prob,a_prob_full).squeeze()

            kl_div_rew = torch.sum(torch.log(a_prob/(a_marginal + 1e-8)) * a_prob , 1)
            cum_kl_rew += kl_div_rew.cpu().numpy()

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
            episode_dict["values"][step] = value_agent

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

            hi_rew = self.config.training.env_rew_coef*agent_reward + (1.0 - fracs[0]) * (self.config.training.kl_rew_coef - 1)* kl_div_rew

            if step == self.config.training.rollout_steps - 1:
                for ind, term in enumerate(switch_ind):
                    if term == 1:
                        hi_ep_dict[ind]["hi_rewards"] = torch.cat((hi_ep_dict[ind]["hi_rewards"], (hi_cum_rew[ind]/curr_p_len[ind]).view(-1)), -1)
                        hi_ep_dict[ind]["hi_rewards"] = torch.cat((hi_ep_dict[ind]["hi_rewards"], hi_rew[ind].view(-1)), -1)
                    else:
                        hi_cum_rew[ind] += hi_rew[ind]
                        hi_ep_dict[ind]["hi_rewards"] = torch.cat((hi_ep_dict[ind]["hi_rewards"], (hi_cum_rew[ind]/(curr_p_len[ind] - next_terminations[ind])).view(-1)), -1)
            else:
                for ind, term in enumerate(switch_ind):
                    if term == 1:
                        if "hi_rewards" not in hi_ep_dict[ind].keys():
                            hi_ep_dict[ind]["hi_rewards"] = hi_cum_rew[ind].view(-1)
                        else:
                            hi_ep_dict[ind]["hi_rewards"] = torch.cat((hi_ep_dict[ind]["hi_rewards"], (hi_cum_rew[ind]/curr_p_len[ind]).view(-1)), -1)
                        hi_cum_rew[ind] = hi_rew[ind]
                        curr_p_len[ind] = next_terminations[ind]
                        switch_ind[ind] -= 1
                    else:
                        hi_cum_rew[ind] += hi_rew[ind]

            next_terminations = next_terminations - 1
        self.initial_obs = next_obs
        

        return episode_dict, hi_ep_dict, info, cum_kl_rew