import os
import gym
import torch
import numpy as np
import torch.nn as nn
import envs
from hydra.utils import instantiate, get_original_cwd


def make_env(gym_id, layout, seed):
    def thunk():
        env = gym.make(gym_id,layout = layout)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def get_partners(config, device, partner_ind = None):
    pop_size = config.training.population_size    
    if partner_ind is None:
        partner_ind = list(range(pop_size))
    train_agent_types = config.training.agent_types
    eval_agent_type = config.evaluation.agent_type
    partners_dict = {
        "train": f"{config.layout.partner_pop_path}",
        "test" : f"{config.layout.eval_partner_pop_path if config.layout.eval_partner_pop_path is not None else config.layout.partner_pop_path}"
    }
    train_partners = []
    test_partners = []
    for agent_ind in range(config.training.population_size):
        for agent_type in train_agent_types:
            agent = instantiate(config.model.partner, config).to(device)

            if "bc" in agent_type:
                agent.load_state_dict(torch.load(get_original_cwd() + "/" + partners_dict["train"] + "/bc_model1.pt", map_location=f'cuda:{config.device_id}'))
            else:
                if "random" not in agent_type:
                    agent.load_state_dict(torch.load(get_original_cwd() + "/" + partners_dict["train"] + f"/agent{agent_ind}_{agent_type}_model.pt", map_location=f'cuda:{config.device_id}')["model_state_dict"])                

            agent.eval()
            train_partners.append(agent)
        agent = instantiate(config.model.partner, config).to(device)

        ###### bc models
        if "bc" in agent_type:
            agent.load_state_dict(torch.load(get_original_cwd() + "/" + partners_dict["train"] + "/bc_model2.pt", map_location=f'cuda:{config.device_id}'))
        else:
            agent.load_state_dict(torch.load(get_original_cwd() + "/" + partners_dict["test"] + f"/agent{agent_ind}_{eval_agent_type}_model.pt", map_location=f'cuda:{config.device_id}')["model_state_dict"])


        agent.eval()
        test_partners.append(agent)
    return train_partners, test_partners


def generate_ep_partners(partners_list, num_envs):
    partner_inds = np.random.randint(len(partners_list),size = num_envs)
    partner_roles = np.random.randint(2, size = num_envs)
    player_roles = np.random.randint(2, size = (num_envs, 1))
    player_roles = np.concatenate((player_roles, 1 -player_roles),1) + 1
    return partner_inds, partner_roles, player_roles