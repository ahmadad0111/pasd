import gym
import numpy as np
import torch
import torch.nn as nn
from runners.utils import compute_advantage, compute_advantage_hi,compute_termination_advantage, SupConReward,  make_multi_view
from utils.utils import generate_ep_partners, make_env

class RIAYNRunner:
    def __init__(self, config, device, **kwargs):
        self.config = config
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(config.gym_id, config.layout.name, config.seed + i) for i in range(self.config.training.num_envs)]
        )
        self.initial_obs = self.envs.reset()
        self.device = device
        self.contreward = SupConReward(temperature=0.5)

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
            "disc_weights": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),

            "disc_emb": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs, 128)).to(self.device),
            "disc_rewards": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            "kl_div_reward": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            

            "termination_logprobs": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            "termination_actions": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs), dtype=torch.int64).to(self.device),
            "termination_values": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            "termination_values_all": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            "termination_advantages": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            "termination_returns": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
            "termination_reward": torch.zeros((self.config.training.rollout_steps, self.config.training.num_envs)).to(self.device),
        }

    def make_episode_hi_dict(self):
        hi_ep = {}
        for env in range(self.config.training.num_envs):
            hi_ep[env] = {}
            hi_ep[env]["inds"] = []
        return hi_ep

    # def generate_episode(self, agent, partner_agents,lstm_state, fracs):
    #     episode_dict = self.make_episode_dict()
    #     hi_ep_dict = self.make_episode_hi_dict()
    #     episode_dict, hi_ep_dict, info, cum_kl_rew = self.rollout(agent, partner_agents, episode_dict, hi_ep_dict, lstm_state, fracs)
    #     for ind in range(self.config.training.num_envs):
    #         hi_ep_dict[ind]["z_adv"] = torch.zeros_like(hi_ep_dict[ind]["hi_rewards"]).to(self.device)
    #     episode_dict = compute_advantage(episode_dict,self.config)
    #     hi_ep_dict = compute_advantage_hi(hi_ep_dict,self.config)
    #     return episode_dict, hi_ep_dict, info, cum_kl_rew


    def generate_episode(self, lo_agent, hi_agent, discriminator, partner_agents, lstm_state_lo, lstm_state_hi, lstm_state_disc, fracs,lam_now,stage):
        episode_dict = self.make_episode_dict()
        hi_ep_dict = self.make_episode_hi_dict()

        if stage == "s1":
            episode_dict, hi_ep_dict, info, cum_kl_rew = self.rollout_s1(lo_agent, discriminator, partner_agents, episode_dict, hi_ep_dict, lstm_state_lo, lstm_state_disc, fracs)
            episode_dict = compute_advantage(episode_dict,self.config)
            return episode_dict, hi_ep_dict, info, cum_kl_rew

        
        elif stage == "s2":

            episode_dict, hi_ep_dict, info, cum_kl_rew = self.rollout(lo_agent,hi_agent, discriminator, partner_agents, episode_dict, hi_ep_dict, lstm_state_lo, lstm_state_disc,lstm_state_hi, fracs)
            for ind in range(self.config.training.num_envs):
                hi_ep_dict[ind]["z_adv"] = torch.zeros_like(hi_ep_dict[ind]["hi_rewards"]).to(self.device)
            episode_dict = compute_advantage(episode_dict,self.config)
            hi_ep_dict = compute_advantage_hi(hi_ep_dict,self.config)
            return episode_dict, hi_ep_dict, info, cum_kl_rew
        
        elif stage == "combined":
            episode_dict, hi_ep_dict, info, cum_kl_rew = self.rollout_combined(lo_agent,hi_agent, discriminator, partner_agents, episode_dict, hi_ep_dict, lstm_state_lo, lstm_state_disc,lstm_state_hi, fracs,lam_now)

            for ind in range(self.config.training.num_envs):
                hi_ep_dict[ind]["z_adv"] = torch.zeros_like(hi_ep_dict[ind]["hi_rewards"]).to(self.device)

                # for low level policy
                paired_feats = make_multi_view(episode_dict["disc_emb"][:,ind,:], episode_dict["current_zs"][:,ind], n_views=2)

                rewards = self.contreward(paired_feats, episode_dict["current_zs"][:,ind])
                
                episode_dict["disc_rewards"][:,ind] = rewards 
                
                #hi_ep_dict[ind]["hi_rewards"] =  (1 - lam_now) * hi_ep_dict[ind]["hi_rewards"] + lam_now * rewards


                # for high level policy reward ########################
                cur_z = None
                accum_reward = 0.0
                step_count = 0.0
                for t in range(len(rewards)):
                    z_t = episode_dict["current_zs"][t, ind].item()
                    r_t = episode_dict["disc_rewards"][t, ind].item()

                    if cur_z is None:
                        cur_z = z_t
                    if z_t == cur_z:
                        accum_reward += r_t
                        step_count += 1
                    else:
                        if step_count > 0:
                            norm_reward = accum_reward / step_count
                        else:
                            norm_reward = accum_reward  # fallback, though step_count=0 shouldn't happen
                        

                        hi_ep_dict[ind]["hi_rewards"] =  (1 - lam_now) * hi_ep_dict[ind]["hi_rewards"] + lam_now * norm_reward
                        cur_z = z_t
                        accum_reward = r_t 
                        step_count = 1
                if cur_z is not None and step_count > 0:
                    norm_reward = accum_reward / step_count
                    hi_ep_dict[ind]["hi_rewards"] = (1 - lam_now) * hi_ep_dict[ind]["hi_rewards"] + lam_now * norm_reward
                #######################################


            # combined_rewards = (1 - lam_now) * episode_dict["rewards"] + lam_now * ( (0.5 * episode_dict["kl_div_reward"]) + (0.5 * episode_dict["disc_rewards"]))
            combined_rewards = (1 - lam_now) * episode_dict["rewards"] + lam_now * ( (1 * episode_dict["kl_div_reward"]) + (0 * episode_dict["disc_rewards"]))
            # Update back
            episode_dict["rewards"] = combined_rewards




            episode_dict = compute_advantage(episode_dict,self.config)
            hi_ep_dict = compute_advantage_hi(hi_ep_dict,self.config)
            return episode_dict, hi_ep_dict, info, cum_kl_rew

        elif stage == "combined_term":
            episode_dict, hi_ep_dict, info, cum_kl_rew = self.rollout_combined_term(lo_agent,hi_agent, discriminator, partner_agents, episode_dict, hi_ep_dict, lstm_state_lo, lstm_state_disc,lstm_state_hi, fracs,lam_now)
            for ind in range(self.config.training.num_envs):
                hi_ep_dict[ind]["z_adv"] = torch.zeros_like(hi_ep_dict[ind]["hi_rewards"]).to(self.device)
            episode_dict = compute_advantage(episode_dict,self.config)
            hi_ep_dict = compute_advantage_hi(hi_ep_dict,self.config)
            episode_dict = compute_termination_advantage(episode_dict,self.config)
            return episode_dict, hi_ep_dict, info, cum_kl_rew
                

    def rollout_s1(self, lo_agent, discriminator, partner_agents, episode_dict, hi_ep_dict, lstm_state_lo, lstm_state_disc, fracs):

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

        current_z = torch.randint(low=0, high=self.config.layout.z_dim, size=(self.config.training.num_envs,), device=self.device)


        # this is for kl_cum_rew
        z_prob = torch.full((self.config.training.num_envs, self.config.layout.z_dim), 1.0 / self.config.layout.z_dim, device=self.device)


        for step in range(self.config.training.rollout_steps):
            episode_dict["dones"][step] = next_done
            current_lstm_state_lo = (lstm_state_lo[0].clone(), lstm_state_lo[1].clone())
            current_lstm_state_disc = (lstm_state_disc[0].clone(), lstm_state_disc[1].clone())

            if step == 0:
                new_p_steps = np.random.randint(self.config.training.p_range[0], self.config.training.p_range[1], size = self.config.training.num_envs)
                next_terminations = new_p_steps
                current_z = torch.zeros((self.config.training.num_envs), dtype = torch.int64).to(self.device)
            else:
                for ind, pstep in enumerate(next_terminations):
                    if pstep == 0:
                        new_p_step = np.random.randint(self.config.training.p_range[0], self.config.training.p_range[1])
                        next_terminations[ind] = new_p_step
                        current_z[ind] = (current_z[ind] + 1) % self.config.layout.z_dim


            # print(f"Current z: {current_z}")
            
            episode_dict["current_zs"][step] = current_z

            pacts_list = []

            with torch.no_grad():
                actions_agent_full, logprob_agent_full, _, value_agent_full, __, a_prob_full, lstm_state_lo = lo_agent.get_action_and_value(obs_agent, expanded_z, next_done, current_lstm_state_lo, expanded = True)
                for ind,partner in enumerate(partners):
                    partner_actions,__, _, ___, ____ = partner_agents[partner].get_action_and_value(obs_list[ind])
                    pacts_list.append(partner_actions)
                
                disc_logits, lstm_state_disc = discriminator.get_disc_logits(obs_agent, next_done, current_lstm_state_disc)


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
            episode_dict["disc_weights"][step] = (agent_reward > 0).float()


            # compute discriminator reward
            log_probs = torch.log_softmax(disc_logits, dim=-1)
            log_q_zs = log_probs[torch.arange(self.config.training.num_envs, device=self.device), current_z]
            log_prior = -torch.log(torch.tensor(self.config.layout.z_dim, dtype=torch.float32, device=self.device))
            intrinsic_reward = (log_q_zs - log_prior)


            # episode_dict["rewards"][step] = agent_reward
            episode_dict["rewards"][step] = intrinsic_reward + agent_reward


            next_terminations = next_terminations - 1
            
            

        self.initial_obs = next_obs
        

        return episode_dict, hi_ep_dict, info, cum_kl_rew

    def rollout_combined_term(self,lo_agent,hi_agent, discriminator, partner_agents, episode_dict, hi_ep_dict, lstm_state_lo, lstm_state_disc,lstm_state_hi, fracs,lam_now):

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
            current_lstm_state_hi = (lstm_state_hi[0].clone(), lstm_state_hi[1].clone())
            current_lstm_state_lo = (lstm_state_lo[0].clone(), lstm_state_lo[1].clone())
            current_lstm_state_disc = (lstm_state_disc[0].clone(), lstm_state_disc[1].clone())
            with torch.no_grad():
                z, logprob_z, _, value_z, lstm_state_hi, z_prob = hi_agent.get_z_and_value(obs_agent, next_done, current_lstm_state_hi)

                # termination we compute for all z's
                term_action_full, term_logprob_full, _, term_value_full, _, term_prob_full = hi_agent.get_termination_and_value(obs_agent, expanded_z, next_done, current_lstm_state_hi, expanded = True)
            term_action_full = term_action_full.reshape(self.config.layout.z_dim, self.config.training.num_envs)
            term_logprob_full = term_logprob_full.reshape(self.config.layout.z_dim, self.config.training.num_envs)
            term_value_full = term_value_full.flatten().reshape(self.config.layout.z_dim, self.config.training.num_envs)
            term_prob_full = term_prob_full.reshape(self.config.layout.z_dim, self.config.training.num_envs, -1)

            term_action = torch.gather(term_action_full, 0,current_z.view(1,-1)).reshape(-1)
            logprob_term = torch.gather(term_logprob_full, 0,current_z.view(1,-1)).reshape(-1)
            value_term = torch.gather(term_value_full, 0,current_z.view(1,-1)).reshape(-1)
            term_prob = torch.gather(term_prob_full, 0, torch.tile(current_z.view(1,-1,1),(1,1,1))).squeeze()



            if step == 0:
                new_p_steps = np.random.randint(1, 2, size = self.config.training.num_envs)
                next_terminations = new_p_steps
                curr_p_len = new_p_steps
                current_z = z.clone()
                #termination
                term_action = torch.zeros(self.config.training.num_envs, dtype=torch.int64).to(self.device)
                hi_option_step_counter = torch.zeros(self.config.training.num_envs, dtype=torch.int32).to(self.device)

                for ind in range(self.config.training.num_envs):
                    hi_ep_dict[ind]["obs"] = obs_agent[ind].view((1,) + tuple(self.config.layout.observation_shape))
                    hi_ep_dict[ind]["z_s"] = z[ind].view(1)
                    hi_ep_dict[ind]["z_logprobs"] = logprob_z[ind].view(1)
                    hi_ep_dict[ind]["z_values"] = value_z[ind]
                    hi_ep_dict[ind]["inds"].append(step)
            else:
                for ind, pstep in enumerate(next_terminations):
                    if pstep == 0:
                        new_p_step = np.random.randint(1, 2)
                        next_terminations[ind] = new_p_step

                        if term_action[ind] == 1:
                            current_z[ind] = z[ind]
                            switch_ind[ind] += 1
                            hi_ep_dict[ind]["obs"] = torch.cat((hi_ep_dict[ind]["obs"],obs_agent[ind].view((1,) + tuple(self.config.layout.observation_shape))),0)
                            hi_ep_dict[ind]["z_s"] = torch.cat((hi_ep_dict[ind]["z_s"],z[ind].view(-1)),-1)
                            hi_ep_dict[ind]["z_logprobs"] = torch.cat((hi_ep_dict[ind]["z_logprobs"],logprob_z[ind].view(-1)),-1)
                            hi_ep_dict[ind]["z_values"] = torch.cat((hi_ep_dict[ind]["z_values"],value_z[ind]),-1)
                            hi_ep_dict[ind]["inds"].append(step)
                        else:
                            curr_p_len += next_terminations[ind]

            episode_dict["current_zs"][step] = current_z


            episode_dict["termination_logprobs"][step] = logprob_term
            episode_dict["termination_actions"][step] = term_action
            episode_dict["termination_values"][step] = value_term
            episode_dict["termination_values_all"][step] = (z_prob * term_value_full.T).sum(dim=1).detach()

            pacts_list = []

            with torch.no_grad():
                actions_agent_full, logprob_agent_full, _, value_agent_full, __, a_prob_full,lstm_state_lo  = lo_agent.get_action_and_value(obs_agent, expanded_z, next_done, current_lstm_state_lo, expanded = True)
                for ind,partner in enumerate(partners):
                    partner_actions,__, _, ___, ____ = partner_agents[partner].get_action_and_value(obs_list[ind])
                    pacts_list.append(partner_actions)
                
                disc_logits, lstm_state_disc = discriminator.get_disc_logits(obs_agent, next_done, current_lstm_state_disc)

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
            

            # compute discriminator reward
            log_probs = torch.log_softmax(disc_logits, dim=-1)
            log_q_zs = log_probs[torch.arange(self.config.training.num_envs, device=self.device), current_z]
            log_prior = -torch.log(torch.tensor(self.config.layout.z_dim, dtype=torch.float32, device=self.device))
            intrinsic_reward = (log_q_zs - log_prior)
            

            # compute theoretical min and max
            K = self.config.layout.z_dim
            r_max = torch.log(torch.tensor(K, device=self.device)) - log_prior  # max intrinsic reward
            r_min = torch.tensor(0.0, device=self.device)                        # min intrinsic reward (skill unidentifiable)

            # normalize
            normalized_intrinsic = (intrinsic_reward - r_min) / (r_max - r_min)
            normalized_intrinsic = torch.clamp(normalized_intrinsic, 0.0, 1.0)  # safety clamp


            term_reward = torch.where(term_action == 1,
                                    1.0 - normalized_intrinsic,   
                                    normalized_intrinsic)  


            # # Normalize intrinsic reward to [0,1]
            # K = self.config.layout.z_dim
            # epsilon = 1e-8
            # r_max = torch.log(torch.tensor(K, device=self.device))
            # r_min = torch.log(torch.tensor(K * epsilon, device=self.device))
            # norm_intrinsic_reward = (intrinsic_reward - r_min) / (r_max - r_min)

            #episode_dict["rewards"][step] = agent_reward #+ intrinsic_reward
            # Combine with environment reward

            #r_low = lam_now * r_intrinsic + (1.0 - lam_now) * r_env
            #episode_dict["rewards"][step] = self.config.training.env_rew_coef * agent_reward #+ self.config.training.disc_rew_coef * normalized_intrinsic


            episode_dict["rewards"][step] = (1 - lam_now) * agent_reward + (lam_now * kl_div_rew)  # intrinsic_reward)

            #episode_dict['termination_reward'][step] = self.config.training.env_rew_coef * agent_reward #+ (1.0 - fracs[0]) * term_reward

            mask = term_action == 1
            # print(f"term_action: {term_action}") 
            # print(f"mask: {mask}")
            episode_dict['termination_reward'][step][mask] = self.config.training.env_rew_coef * agent_reward[mask].float()

            
            hi_rew = self.config.training.env_rew_coef*agent_reward #+ (1.0 - fracs[0]) * (self.config.training.kl_rew_coef - 1)* kl_div_rew
            #hi_rew = agent_reward

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




    def rollout(self,lo_agent,hi_agent, discriminator, partner_agents, episode_dict, hi_ep_dict, lstm_state_lo, lstm_state_disc,lstm_state_hi, fracs):

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
            current_lstm_state_hi = (lstm_state_hi[0].clone(), lstm_state_hi[1].clone())
            current_lstm_state_lo = (lstm_state_lo[0].clone(), lstm_state_lo[1].clone())
            current_lstm_state_disc = (lstm_state_disc[0].clone(), lstm_state_disc[1].clone())
            with torch.no_grad():
                z, logprob_z, _, value_z, lstm_state_hi, z_prob = hi_agent.get_z_and_value(obs_agent, next_done, current_lstm_state_hi)

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
                actions_agent_full, logprob_agent_full, _, value_agent_full, __, a_prob_full,lstm_state_lo  = lo_agent.get_action_and_value(obs_agent, expanded_z, next_done, current_lstm_state_lo, expanded = True)
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

            #hi_rew = self.config.training.env_rew_coef*agent_reward + (1.0 - fracs[0]) * (self.config.training.kl_rew_coef - 1)* kl_div_rew
            hi_rew = agent_reward

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
    


    def rollout_combined(self,lo_agent,hi_agent, discriminator, partner_agents, episode_dict, hi_ep_dict, lstm_state_lo, lstm_state_disc,lstm_state_hi, fracs,lam_now):

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
            current_lstm_state_hi = (lstm_state_hi[0].clone(), lstm_state_hi[1].clone())
            current_lstm_state_lo = (lstm_state_lo[0].clone(), lstm_state_lo[1].clone())
            current_lstm_state_disc = (lstm_state_disc[0].clone(), lstm_state_disc[1].clone())
            with torch.no_grad():
                z, logprob_z, _, value_z, lstm_state_hi, z_prob = hi_agent.get_z_and_value(obs_agent, next_done, current_lstm_state_hi)
                hi_emb = hi_agent.get_high_feat(obs_agent, next_done, current_lstm_state_hi)



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
                    #hi_ep_dict[ind]["hi_emb"] = hi_emb[ind].view(1, -1)

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
                        #hi_ep_dict[ind]["hi_emb"] = torch.cat((hi_ep_dict[ind]["hi_emb"], hi_emb[ind].view(1, -1)), dim=0)


            episode_dict["current_zs"][step] = current_z

            pacts_list = []

            with torch.no_grad():
                actions_agent_full, logprob_agent_full, _, value_agent_full, __, a_prob_full,lstm_state_lo  = lo_agent.get_action_and_value(obs_agent, expanded_z, next_done, current_lstm_state_lo, expanded = True)
                for ind,partner in enumerate(partners):
                    partner_actions,__, _, ___, ____ = partner_agents[partner].get_action_and_value(obs_list[ind])
                    pacts_list.append(partner_actions)
                
                disc_emb, lstm_state_disc = discriminator.get_disc_emb(obs_agent, next_done, current_lstm_state_disc)

                

            episode_dict["disc_emb"][step] = hi_emb  #disc_emb

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



            episode_dict["rewards"][step] = (1 - lam_now) * agent_reward + (lam_now * kl_div_rew)  # intrinsic_reward)
            #episode_dict["rewards"][step] = agent_reward + (1.0 - fracs[0]) * (self.config.training.kl_rew_coef - 1)* kl_div_rew

            #episode_dict["rewards"][step] =  agent_reward 
            episode_dict["kl_div_reward"][step] = kl_div_rew


            # Combine with environment reward
            #episode_dict["rewards"][step] = self.config.training.env_rew_coef * agent_reward + self.config.training.disc_rew_coef * norm_intrinsic_reward



            hi_rew = self.config.training.env_rew_coef*agent_reward + (1.0 - fracs[0]) * (self.config.training.kl_rew_coef - 1)* kl_div_rew
            #hi_rew = agent_reward

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