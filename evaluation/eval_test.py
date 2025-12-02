import os
import gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from tqdm import tqdm
# from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
# from trainers.hipt_trainer import HiPTTrainer
# from trainers.diayn_trainer import DIAYNTrainer
# from trainers.hipt_dynamic_trainer import HiPTTrainer_dyn
from utils.utils import make_env

# from trainers.riayn_trainer import RIAYNTrainer


from modules.torch_agent import infer_hipt, infer_pasd, infer_pop

layout_name_map = {
        "cramped_room": "Cramped Room",
        "asymmetric_advantages": "Asymmetric Advantages",
        "counter_circuit_o_1order": "Counter Circuit",
        "forced_coordination": "Forced Coordination",
        "coordination_ring": "Coordination Ring",
    }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif',fps = 60):

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=fps)
    anim.save(path + filename, writer='imagemagick', fps=fps)

    plt.close()


def plot_sp_heatmap():
    envs = gym.vector.SyncVectorEnv(
        [make_env('Overcooked-v1', 'cramped_room', 30)]
    )
    initial_obs = envs.reset()

    playerZero = 'PPOCrampedRoom'
    # trainer.load_population(config.evaluation.agent_type,path)

    if playerZero == 'PPOCrampedRoom':
        obs_shape = [26,5,4]
        z_dim = 6
    elif playerZero == 'PPOForcedCoordination':
        obs_shape = [26, 5, 5]
        z_dim = 5
    elif playerZero == 'PPOCounterCircuit':
        obs_shape = [26, 8, 5]
        z_dim = 6
    elif playerZero == 'PPOCoordinationRing':
        obs_shape = [26, 5, 5]
        z_dim = 6
    elif playerZero == 'PPOAsymmetricAdvantages':
        obs_shape = [26, 9, 5]
        z_dim = 6

    agent0 = infer_pop(agent_name=playerZero,obs_shape= obs_shape, z_dim= z_dim)
    agent1 = infer_pop(agent_name=playerZero,obs_shape= obs_shape, z_dim= z_dim)

    agent0 = agent0.to(device = device)
    agent1 = agent1.to(device = device)


    next_obs_p0 = torch.Tensor(initial_obs[:,0]).to(device = device)
    next_obs_p1 = torch.Tensor(initial_obs[:,1]).to(device = device)

    rewards = 0
    for step in tqdm(range(400)):
        agent0_actions,__, _, ___, ____ = agent0.get_action_and_value(next_obs_p0)
        agent1_actions,__, _, ___, ____ = agent1.get_action_and_value(next_obs_p1)

        # print(agent0_actions)
        joint_action = torch.cat((agent0_actions.view(1,1),agent1_actions.view(1,1)),1)
        joint_action = joint_action.type(torch.int8)
        next_obs, reward, done, info = envs.step(joint_action.cpu().numpy())

        # print(f"after : shape : {torch.Tensor(next_obs).shape}")



        rewards += reward

        next_obs_p0 = torch.Tensor(next_obs[:,0]).to(device = device)
        next_obs_p1 = torch.Tensor(next_obs[:,1]).to(device = device)

    print(f"Total rewards: {rewards}")
if __name__ == "__main__":
    plot_sp_heatmap()