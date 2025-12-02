
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import gym


import matplotlib.pyplot as plt
import time
import torch
import numpy as np


import os
import gym
import torch
import numpy as np
import torch.nn as nn
import envs
from hydra.utils import instantiate, get_original_cwd


def make_env(gym_id, layout, seed):

    env = gym.make(gym_id,layout = layout)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def visualize_policy_live(env, model_agent0, model_agent1, episodes=1):
    for ep in range(episodes):
        obs = env.reset()  # obs is np.ndarray now
        done = False
        total_reward = 0

        while not done:
            # Here you must split obs between agents:
            # For example, if obs shape is (2, obs_dim) or (obs_dim*2,), adjust accordingly
            # Assuming obs is shape (2, obs_dim):
            obs_agent_0 = obs[0]
            obs_agent_1 = obs[1]

            obs_0_tensor = torch.tensor(obs_agent_0, dtype=torch.float32).unsqueeze(0)
            obs_1_tensor = torch.tensor(obs_agent_1, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                probs_agent_0, _ = model_agent0(obs_0_tensor)
                probs_agent_1, _ = model_agent1(obs_1_tensor)

            action_agent_0 = torch.argmax(probs_agent_0, dim=1).item()
            action_agent_1 = torch.argmax(probs_agent_1, dim=1).item()

            obs, reward, done, info = env.step([action_agent_0, action_agent_1])

            
            total_reward += reward

            env.render()

            time.sleep(0.05)

        print(f"Episode {ep + 1} finished with total reward: {total_reward}")

    env.close()


def dummy_models():
    class DummyModel:
        def __call__(self, x):
            batch_size = x.shape[0]
            # Return uniform random logits for 5 actions + dummy second output
            return torch.rand(batch_size, 5), None
    return DummyModel(), DummyModel()

model_agent0, model_agent1 = dummy_models()


env = make_env(gym_id= "Overcooked-v1", layout= "counter_circuit_o_1order", seed= 10)


visualize_policy_live(env, model_agent0, model_agent1, episodes=1)



python main.py model=population layout=counter_circuit && python main.py model=population layout=coordination_ring && python main.py model=population layout=forced_coordination
