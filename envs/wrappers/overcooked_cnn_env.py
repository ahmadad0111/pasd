import gym
from gym import error, spaces, utils
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.visualization.pygame_utils import MultiFramePygameImage
from overcooked_ai_py.visualization.visualization_utils import show_ipython_images_slider
import numpy as np
import sys
# import torch
import overcooked_ai_py.mdp.overcooked_env as oce
import overcooked_ai_py.mdp.overcooked_mdp as omdp
print("Imported from")
print(oce.__file__)   # path to overcooked_env.py
print(omdp.__file__)  # path to overcooked_mdp.py
# C:\Users\adnan\overcooked_ai\src\overcooked_ai_py\mdp

class OvercookedEnvWrapper(gym.Env):
    env_name = "Overcooked-v1"

    def __init__(self, layout, horizon = 400):
        super(OvercookedEnvWrapper, self).__init__()

        self.layout = layout
        self.horizon = horizon
        self.visualizer = StateVisualizer()
        self.agent_roles = None

        self.mdp = OvercookedGridworld.from_layout_name(self.layout)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon = self.horizon, info_level=0)
        self.featurize_fn = self.base_env.lossless_state_encoding_mdp
        self.observation_space = self._setup_observation_space()
        self.action_space = gym.spaces.MultiDiscrete([len(Action.ALL_ACTIONS),len(Action.ALL_ACTIONS)])

    def _setup_observation_space(self):
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_shape = (2,) + self.featurize_fn(dummy_state)[0].shape[-1:] + self.featurize_fn(dummy_state)[0].shape[0:2]
        high = np.ones(obs_shape) * float("inf")
        low = np.zeros(obs_shape)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def set_agent_roles(self, p0 ,p1):
        if self.agent_roles is None:
            self.agent_roles = {"p0" : p0, "p1" : p1}
        else:
            self.agent_roles["p0"] = p0
            self.agent_roles["p1"] = p1

    def step(self, action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        # assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid"%(action, type(action))
        # print(action)
        agent_action, other_agent_action = [Action.INDEX_TO_ACTION[a] for a in action]

        joint_action = (agent_action, other_agent_action)

        if self.agent_roles is None:
            self.agent_roles = {"p0" : 0, "p1" : 0}

        next_state, reward, done, env_info = self.base_env.step(joint_action, agent_roles = self.agent_roles)
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        #print(f"after : shape : {torch.Tensor(next_state).shape}")


        ob_p0 = np.reshape(ob_p0, (1,) + (self.observation_space.shape[1:]))
        ob_p1 = np.reshape(ob_p1, (1,) + (self.observation_space.shape[1:]))
        obs = np.concatenate((ob_p0, ob_p1))

        shaped_reward = sum(env_info["shaped_r_by_agent"])

        return obs, reward, done, env_info

    def reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        # self.mdp = self.base_env.mdp
        # self.agent_idx = np.random.choice([0, 1])
        ob_p0, ob_p1 = self.featurize_fn( self.base_env.state)

        ob_p0 = np.reshape(ob_p0, (1,) + (self.observation_space.shape[1:]))
        ob_p1 = np.reshape(ob_p1, (1,) + (self.observation_space.shape[1:]))


        # if self.agent_idx == 0:
        #     both_agents_ob = (ob_p0, ob_p1)
        # else:
        #     both_agents_ob = (ob_p1, ob_p0)

        # agents_obs = (ob_p0, ob_p1)
        agents_obs = np.concatenate((ob_p0, ob_p1))

        return agents_obs


    def render(self, mode = 'human'):
        self.visualizer.display_rendered_state(self.base_env.state, grid = self.mdp.terrain_mtx, ipython_display=True, window_display=True)
