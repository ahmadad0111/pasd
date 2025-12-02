from human_aware_rl.imitation.behavior_cloning_tf2 import _get_base_ae, BehaviorCloningPolicy
from human_aware_rl.imitation.behavior_cloning_tf2 import load_bc_model
import tensorflow as tf
from tensorflow import keras
from human_aware_rl.rllib.rllib import AgentPair
from human_aware_rl.rllib.rllib import RlLibAgent
# from pylsl import StreamInfo, StreamOutlet
import json
import numpy as np
import os
import pickle
import gym
import random
from abc import ABC, abstractmethod
from queue import Empty, Full, LifoQueue, Queue
from threading import Lock, Thread
from datetime import datetime, timezone
from time import time

import ray
from utils import DOCKER_VOLUME, create_dirs

from human_aware_rl.rllib.rllib import load_agent
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MotionPlanner,
)

from modules.torch_agent import infer_hipt, infer_pasd, infer_pop
import torch

from database import Database
# from database1 import Database1
from keyboard_tracker import KeyboardTracker
from keyboard_tracking_controller import TrackingController 


import uuid
import hashlib

# Read in global config
CONF_PATH = os.getenv("CONF_PATH", "config.json")
with open(CONF_PATH, "r") as f:
    CONFIG = json.load(f)

def generate_unique_hash():
    user_id = "user123"
    timestamp = str(datetime.now())
    random_string = str(uuid.uuid4())
    session_data = user_id + timestamp + random_string
    return hashlib.sha256(session_data.encode()).hexdigest()


database = Database() 
# database1 = Database1()

# Relative path to where all static pre-trained agents are stored on server
AGENT_DIR = None

# Maximum allowable game time (in seconds)
MAX_GAME_TIME = None


def _configure(max_game_time, agent_dir):
    global AGENT_DIR, MAX_GAME_TIME
    MAX_GAME_TIME = max_game_time
    AGENT_DIR = agent_dir


def fix_bc_path(path):
    """
    Loading a PPO agent trained with a BC agent requires loading the BC model as well when restoring the trainer, even though the BC model is not used in game
    For now the solution is to include the saved BC model and fix the relative path to the model in the config.pkl file
    """

    import dill

    # the path is the agents/Rllib.*/agent directory
    agent_path = os.path.dirname(path)
    with open(os.path.join(agent_path, "config.pkl"), "rb") as f:
        data = dill.load(f)
    bc_model_dir = data["bc_params"]["bc_config"]["model_dir"]
    last_dir = os.path.basename(bc_model_dir)
    bc_model_dir = os.path.join(agent_path, "bc_params", last_dir)
    data["bc_params"]["bc_config"]["model_dir"] = bc_model_dir
    with open(os.path.join(agent_path, "config.pkl"), "wb") as f:
        dill.dump(data, f)


class Game(ABC):

    """
    Class representing a game object. Coordinates the simultaneous actions of arbitrary
    number of players. Override this base class in order to use.

    Players can post actions to a `pending_actions` queue, and driver code can call `tick` to apply these actions.


    It should be noted that most operations in this class are not on their own thread safe. Thus, client code should
    acquire `self.lock` before making any modifications to the instance.

    One important exception to the above rule is `enqueue_actions` which is thread safe out of the box
    """

    # Possible TODO: create a static list of IDs used by the class so far to verify id uniqueness
    # This would need to be serialized, however, which might cause too great a performance hit to
    # be worth it

    EMPTY = "EMPTY"

    class Status:
        DONE = "done"
        ACTIVE = "active"
        RESET = "reset"
        INACTIVE = "inactive"
        ERROR = "error"

    def __init__(self, *args, **kwargs):
        """
        players (list): List of IDs of players currently in the game
        spectators (set): Collection of IDs of players that are not allowed to enqueue actions but are currently watching the game
        id (int):   Unique identifier for this game
        pending_actions List[(Queue)]: Buffer of (player_id, action) pairs have submitted that haven't been commited yet
        lock (Lock):    Used to serialize updates to the game state
        is_active(bool): Whether the game is currently being played or not
        """
        self.players = []
        self.spectators = set()
        self.pending_actions = []
        self.id = kwargs.get("id", id(self))
        self.lock = Lock()
        self._is_active = False
        self.xai_explanation = ''

    @abstractmethod
    def is_full(self):
        """
        Returns whether there is room for additional players to join or not
        """
        pass

    @abstractmethod
    def apply_action(self, player_idx, action):
        """
        Updates the game state by applying a single (player_idx, action) tuple. Subclasses should try to override this method
        if possible
        """
        pass

    @abstractmethod
    def is_finished(self):
        """
        Returns whether the game has concluded or not
        """
        pass

    def is_ready(self):
        """
        Returns whether the game can be started. Defaults to having enough players
        """
        return self.is_full()

    @property
    def is_active(self):
        """
        Whether the game is currently being played
        """
        return self._is_active

    @property
    def reset_timeout(self):
        """
        Number of milliseconds to pause game on reset
        """
        return 3000

    def apply_actions(self):
        """
        Updates the game state by applying each of the pending actions in the buffer. Is called by the tick method. Subclasses
        should override this method if joint actions are necessary. If actions can be serialized, overriding `apply_action` is
        preferred
        """
        for i in range(len(self.players)):
            try:
                while True:
                    action = self.pending_actions[i].get(block=False)
                    self.apply_action(i, action)
            except Empty:
                pass

    def activate(self):
        """
        Activates the game to let server know real-time updates should start. Provides little functionality but useful as
        a check for debugging
        """
        self._is_active = True

    def deactivate(self):
        """
        Deactives the game such that subsequent calls to `tick` will be no-ops. Used to handle case where game ends but
        there is still a buffer of client pings to handle
        """
        self._is_active = False

    def reset(self):
        """
        Restarts the game while keeping all active players by resetting game stats and temporarily disabling `tick`
        """
        if not self.is_active:
            raise ValueError("Inactive Games cannot be reset")
        if self.is_finished():
            return self.Status.DONE
        self.deactivate()
        self.activate()
        return self.Status.RESET

    def needs_reset(self):
        """
        Returns whether the game should be reset on the next call to `tick`
        """
        return False

    def tick(self):
        """
        Updates the game state by applying each of the pending actions. This is done so that players cannot directly modify
        the game state, offering an additional level of safety and thread security.

        One can think of "enqueue_action" like calling "git add" and "tick" like calling "git commit"

        Subclasses should try to override `apply_actions` if possible. Only override this method if necessary
        """
        if not self.is_active:
            return self.Status.INACTIVE
        if self.needs_reset():
            self.reset()
            return self.Status.RESET

        self.apply_actions()
        return self.Status.DONE if self.is_finished() else self.Status.ACTIVE

    def enqueue_action(self, player_id, action):
        """
        Add (player_id, action) pair to the pending action queue, without modifying underlying game state

        Note: This function IS thread safe
        """
        if not self.is_active:
            # Could run into issues with is_active not being thread safe
            return
        if player_id not in self.players:
            # Only players actively in game are allowed to enqueue actions
            return
        try:
            player_idx = self.players.index(player_id)
            self.pending_actions[player_idx].put(action)
        except Full:
            pass

    def update_explanation(self, explanation):
        self.xai_explanation = explanation

    def get_state(self):
        """
        Return a JSON compatible serialized state of the game. Note that this should be as minimalistic as possible
        as the size of the game state will be the most important factor in game performance. This is sent to the client
        every frame update.
        """
        return {"players": self.players}

    def to_json(self):
        """
        Return a JSON compatible serialized state of the game. Contains all information about the game, does not need to
        be minimalistic. This is sent to the client only once, upon game creation
        """
        return self.get_state()

    def is_empty(self):
        """
        Return whether it is safe to garbage collect this game instance
        """
        return not self.num_players

    def add_player(self, player_id, idx=None, buff_size=-1):
        """
        Add player_id to the game
        """
        if self.is_full():
            raise ValueError("Cannot add players to full game")
        if self.is_active:
            raise ValueError("Cannot add players to active games")
        if not idx and self.EMPTY in self.players:
            idx = self.players.index(self.EMPTY)
        elif not idx:
            idx = len(self.players)

        padding = max(0, idx - len(self.players) + 1)
        for _ in range(padding):
            self.players.append(self.EMPTY)
            self.pending_actions.append(self.EMPTY)

        self.players[idx] = player_id
        self.pending_actions[idx] = Queue(maxsize=buff_size)

    def add_spectator(self, spectator_id):
        """
        Add spectator_id to list of spectators for this game
        """
        if spectator_id in self.players:
            raise ValueError("Cannot spectate and play at same time")
        self.spectators.add(spectator_id)

    def remove_player(self, player_id):
        """
        Remove player_id from the game
        """
        try:
            idx = self.players.index(player_id)
            self.players[idx] = self.EMPTY
            self.pending_actions[idx] = self.EMPTY
        except ValueError:
            return False
        else:
            return True

    def remove_spectator(self, spectator_id):
        """
        Removes spectator_id if they are in list of spectators. Returns True if spectator successfully removed, False otherwise
        """
        try:
            self.spectators.remove(spectator_id)
        except ValueError:
            return False
        else:
            return True

    def clear_pending_actions(self):
        """
        Remove all queued actions for all players
        """
        for i, player in enumerate(self.players):
            if player != self.EMPTY:
                queue = self.pending_actions[i]
                queue.queue.clear()

    @property
    def num_players(self):
        return len([player for player in self.players if player != self.EMPTY])

    def get_data(self):
        """
        Return any game metadata to server driver.
        """
        return {}


class DummyGame(Game):

    """
    Standin class used to test basic server logic
    """

    def __init__(self, **kwargs):
        super(DummyGame, self).__init__(**kwargs)
        self.counter = 0

    def is_full(self):
        return self.num_players == 2

    def apply_action(self, idx, action):
        pass

    def apply_actions(self):
        self.counter += 1

    def is_finished(self):
        return self.counter >= 100

    def get_state(self):
        state = super(DummyGame, self).get_state()
        state["count"] = self.counter
        return state


class DummyInteractiveGame(Game):

    """
    Standing class used to test interactive components of the server logic
    """

    def __init__(self, **kwargs):
        super(DummyInteractiveGame, self).__init__(**kwargs)
        self.max_players = int(
            kwargs.get("playerZero", "human") == "human"
        ) + int(kwargs.get("playerOne", "human") == "human")
        self.max_count = kwargs.get("max_count", 30)
        self.counter = 0
        self.counts = [0] * self.max_players

    def is_full(self):
        return self.num_players == self.max_players

    def is_finished(self):
        return max(self.counts) >= self.max_count

    def apply_action(self, player_idx, action):
        if action.upper() == Direction.NORTH:
            self.counts[player_idx] += 1
        if action.upper() == Direction.SOUTH:
            self.counts[player_idx] -= 1

    def apply_actions(self):
        super(DummyInteractiveGame, self).apply_actions()
        self.counter += 1

    def get_state(self):
        state = super(DummyInteractiveGame, self).get_state()
        state["count"] = self.counter
        for i in range(self.num_players):
            state["player_{}_count".format(i)] = self.counts[i]
        return state


class OvercookedGame(Game):
    """
    Class for bridging the gap between Overcooked_Env and the Game interface

    Instance variable:
        - max_players (int): Maximum number of players that can be in the game at once
        - mdp (OvercookedGridworld): Controls the underlying Overcooked game logic
        - score (int): Current reward acheived by all players
        - max_time (int): Number of seconds the game should last
        - npc_policies (dict): Maps user_id to policy (Agent) for each AI player
        - npc_state_queues (dict): Mapping of NPC user_ids to LIFO queues for the policy to process
        - curr_tick (int): How many times the game server has called this instance's `tick` method
        - ticker_per_ai_action (int): How many frames should pass in between NPC policy forward passes.
            Note that this is a lower bound; if the policy is computationally expensive the actual frames
            per forward pass can be higher
        - action_to_overcooked_action (dict): Maps action names returned by client to action names used by OvercookedGridworld
            Note that this is an instance variable and not a static variable for efficiency reasons
        - human_players (set(str)): Collection of all player IDs that correspond to humans
        - npc_players (set(str)): Collection of all player IDs that correspond to AI
        - randomized (boolean): Whether the order of the layouts should be randomized

    Methods:
        - npc_policy_consumer: Background process that asynchronously computes NPC policy forward passes. One thread
            spawned for each NPC
        - _curr_game_over: Determines whether the game on the current mdp has ended
    """
    uid_value = None
    def __init__(
        self,
        layouts=["cramped_room"],
        mdp_params={},
        num_players=2,
        gameTime=80,
        playerZero="human",
        playerOne="human",
        showPotential=False,
        randomized=False,
        ticks_per_ai_action=1,
        current_phase=1,
        current_round=1,
        current_session=1,
        total_rounds=1,
        **kwargs
    ):
        super(OvercookedGame, self).__init__(**kwargs)
        self.show_potential = showPotential
        self.mdp_params = mdp_params
        self.layouts = layouts
        self.max_players = int(num_players)
        self.mdp = None
        self.mp = None
        self.score = 0
        self.phi = 0
        self.max_time = min(int(gameTime), MAX_GAME_TIME)
        self.npc_policies = {}
        self.npc_state_queues = {}
        self.action_to_overcooked_action = {
            "STAY": Action.STAY,
            "UP": Direction.NORTH,
            "DOWN": Direction.SOUTH,
            "LEFT": Direction.WEST,
            "RIGHT": Direction.EAST,
            "SPACE": Action.INTERACT,
        }
        self.ticks_per_ai_action = ticks_per_ai_action
        self.curr_tick = 0
        self.human_players = set()
        self.npc_players = set()
        self.num_collisions = 0
        self.z = 0

        #self.kb_controller = TrackingController() 

        self.current_phase = current_phase
        self.current_round = current_round
        self.current_session = current_session
        self.total_rounds = total_rounds
        self.xai_explanation = 'test'
        
        
        self.npc_agent = {}
        self.npc_agent_hi = {}
        self.npc_agent_lo = {}
        self.npc_lstm_state = {}
        self.npc_lstm_state_hi = {}
        self.npc_lstm_state_lo = {}
        
        self.agent_type = {}
        self.counter = 1
        self.p0 = ''
        self.p1 = ''


        self.next_done = torch.zeros(1)  
        if(kwargs.get("ai_agent_assignment", []) != []):
            ai_agent_assignment = kwargs.get("ai_agent_assignment", [])
        self.phase_agent_type = ai_agent_assignment[self.current_phase-1]
        print(f"===== Phase  {self.current_phase} agent_type: {self.phase_agent_type}")
        print(f"===== Player zero  {playerZero} Player One: {playerOne}")
        self.p0 = playerZero
        self.p1 = playerOne
        if randomized:
            random.shuffle(self.layouts)

        if playerZero != "human":
            player_zero_id = playerZero + "_0"
            self.add_player(player_zero_id, idx=0, buff_size=1, is_human=False)
            self.npc_policies[player_zero_id] = self.get_policy(
                playerZero, idx=0
            )
            self.agent_type[player_zero_id] = self.phase_agent_type
            self.npc_state_queues[player_zero_id] = LifoQueue()
            print(f" selected player :  {playerZero}")
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
            elif playerZero == 'TutorialAI':
                obs_shape = [26, 5, 5]
                z_dim = 6
            
            if playerZero != 'TutorialAI':

                if self.agent_type[player_zero_id] == 'hipt':
                    self.agent0 = infer_hipt(agent_name=playerZero,obs_shape= obs_shape, z_dim= z_dim)
                    self.lstm_state = (
                    torch.zeros(self.agent0.lstm.num_layers, 1, self.agent0.lstm.hidden_size),
                    torch.zeros(self.agent0.lstm.num_layers, 1, self.agent0.lstm.hidden_size),)   

                    self.npc_agent[player_zero_id] = self.agent0
                    self.npc_lstm_state[player_zero_id]= self.lstm_state
                elif self.agent_type[player_zero_id] == 'pasd':
                    self.agent0_hi, self.agent0_lo =  infer_pasd(agent_name=playerZero,obs_shape= obs_shape, z_dim= z_dim)

                    self.lstm_state_lo = (
                        torch.zeros(self.agent0_lo.lstm.num_layers, 1, self.agent0_lo.lstm.hidden_size),
                        torch.zeros(self.agent0_lo.lstm.num_layers, 1, self.agent0_lo.lstm.hidden_size),
                    )   
                    self.lstm_state_hi = (
                        torch.zeros(self.agent0_hi.lstm.num_layers, 1, self.agent0_hi.lstm.hidden_size),
                        torch.zeros(self.agent0_hi.lstm.num_layers, 1, self.agent0_hi.lstm.hidden_size),
                    )
                    self.npc_agent_hi[player_zero_id] = self.agent0_hi
                    self.npc_agent_lo[player_zero_id] = self.agent0_lo
                    self.npc_lstm_state_lo[player_zero_id]= self.lstm_state_lo
                    self.npc_lstm_state_hi[player_zero_id]= self.lstm_state_hi

                else:
                    self.agent0 = infer_pop(agent_name=playerZero,obs_shape= obs_shape, z_dim= z_dim)
                    self.npc_agent[player_zero_id] = self.agent0
                    print("pop agent loaded")


        if playerOne != "human":

            player_one_id = playerOne + "_1"
            self.agent_type[player_one_id] = self.phase_agent_type
            self.add_player(player_one_id, idx=1, buff_size=1, is_human=False)
            self.npc_policies[player_one_id] = self.get_policy(
                playerOne, idx=1
            )
            self.npc_state_queues[player_one_id] = LifoQueue()
            if playerOne == 'PPOCrampedRoom':
                obs_shape = [26,5,4]
                z_dim = 6
            elif playerOne == 'PPOForcedCoordination':
                obs_shape = [26, 5, 5]
                z_dim = 5
            elif playerOne == 'PPOCounterCircuit':
                obs_shape = [26, 8, 5]
                z_dim = 6
            elif playerOne == 'PPOCoordinationRing':
                obs_shape = [26, 5, 5]
                z_dim = 6
            elif playerOne == 'PPOAsymmetricAdvantages':
                obs_shape = [26, 9, 5]
                z_dim = 6
            elif playerOne == 'TutorialAI':
                obs_shape = [26, 5, 5]
                z_dim = 6

            if playerOne != 'TutorialAI':

                if self.agent_type[player_one_id] == 'hipt':
                    
                    self.agent1 = infer_hipt(agent_name=playerOne,obs_shape= obs_shape, z_dim= z_dim)
                    self.lstm_state = (
                    torch.zeros(self.agent1.lstm.num_layers, 1, self.agent1.lstm.hidden_size),
                    torch.zeros(self.agent1.lstm.num_layers, 1, self.agent1.lstm.hidden_size),)   

                    self.npc_agent[player_one_id] = self.agent1
                    self.npc_lstm_state[player_one_id]= self.lstm_state
                    print("hipt agent loaded")
                elif self.agent_type[player_one_id] == 'pasd':
                    self.agent1_hi, self.agent1_lo =  infer_pasd(agent_name=playerOne,obs_shape= obs_shape, z_dim= z_dim)

                    self.lstm_state_lo = (
                        torch.zeros(self.agent1_lo.lstm.num_layers, 1, self.agent1_lo.lstm.hidden_size),
                        torch.zeros(self.agent1_lo.lstm.num_layers, 1, self.agent1_lo.lstm.hidden_size),
                    )   
                    self.lstm_state_hi = (
                        torch.zeros(self.agent1_hi.lstm.num_layers, 1, self.agent1_hi.lstm.hidden_size),
                        torch.zeros(self.agent1_hi.lstm.num_layers, 1, self.agent1_hi.lstm.hidden_size),
                    )
                    self.npc_agent_hi[player_one_id] = self.agent1_hi
                    self.npc_agent_lo[player_one_id] = self.agent1_lo
                    self.npc_lstm_state_lo[player_one_id]= self.lstm_state_lo
                    self.npc_lstm_state_hi[player_one_id]= self.lstm_state_hi
                    print("pasd agent loaded")
                else:
                    self.agent1 = infer_pop(agent_name=playerOne,obs_shape= obs_shape, z_dim= z_dim)
                    self.npc_agent[player_one_id] = self.agent1
                    print("pop agent loaded")
        # Always kill ray after loading agent, otherwise, ray will crash once process exits
        # Only kill ray after loading both agents to avoid having to restart ray during loading
        if ray.is_initialized():
            ray.shutdown()

        if kwargs["dataCollection"]:
            self.write_data = True
            self.write_config = kwargs["collection_config"]
        else:
            self.write_data = False

        self.trajectory = []
    

    # def set_uid(self, uid_value):
    #     self.uid = uid_value
    @classmethod
    def set_uid(cls, uid_value):
        cls.uid_value = uid_value
        print(f"Setting UID: {uid_value}")

    @classmethod
    def get_uid(cls):
        return cls.uid_value

    # def start_recording_kb_events(self, session_id):
    #     self.kb_controller.start_tracking(session_id) 
        
   

    # def stop_recording_kb_events(self):
    #     self.kb_controller.stop_tracking() 


        
    def _curr_game_over(self):

        return time() - self.start_time >= self.max_time
    
    def needs_reset(self):
        return self._curr_game_over() and not self.is_finished()

    def add_player(self, player_id, idx=None, buff_size=-1, is_human=True):
        super(OvercookedGame, self).add_player(
            player_id, idx=idx, buff_size=buff_size
        )
        if is_human:
            self.human_players.add(player_id)
        else:
            self.npc_players.add(player_id)

    def remove_player(self, player_id):
        removed = super(OvercookedGame, self).remove_player(player_id)
        if removed:
            if player_id in self.human_players:
                self.human_players.remove(player_id)
            elif player_id in self.npc_players:
                self.npc_players.remove(player_id)
            else:
                raise ValueError("Inconsistent state")
            
    def _setup_observation_space(self, mdp):
        dummy_state = mdp.get_standard_start_state()
        obs_shape = (2,) + mdp.lossless_state_encoding(dummy_state)[0].shape[-1:] + mdp.lossless_state_encoding(dummy_state)[0].shape[0:2]
        high = np.ones(obs_shape) * float("inf")
        low = np.zeros(obs_shape)
        return gym.spaces.Box(low, high, dtype=np.float32)
    
    def npc_policy_consumer(self, policy_id, mdp):
        queue = self.npc_state_queues[policy_id]
        policy = self.npc_policies[policy_id]
        while self._is_active:
            state = queue.get()
            
            # get state encoding
            self.observation_space = self._setup_observation_space(mdp)
            ob_p0, ob_p1 = mdp.lossless_state_encoding(state)
            ob_p0 = np.reshape(ob_p0, (1,) + (self.observation_space.shape[1:]))
            ob_p1 = np.reshape(ob_p1, (1,) + (self.observation_space.shape[1:]))
            obs = np.concatenate((ob_p0, ob_p1))
            my_obs = obs[int(policy_id.split('_')[-1])]
            obs = torch.tensor(my_obs, dtype=torch.float32)
            obs = obs.unsqueeze(0)

            if self.p0 != 'TutorialAI' and self.p1 != 'TutorialAI':
                with torch.no_grad():
                    if self.agent_type[policy_id] == 'pasd':
                        z, _, ent, ___, self.npc_lstm_state_hi[policy_id], ____ = self.npc_agent_hi[policy_id].get_z_and_value(obs, self.next_done, self.npc_lstm_state_hi[policy_id])
                        self.z = z

                        agent_action, _, __, ___,_, probs, self.npc_lstm_state_lo[policy_id] = self.npc_agent_lo[policy_id].get_action_and_value(obs, z, self.next_done, self.npc_lstm_state_lo[policy_id])


                        
                        npc_action = Action.INDEX_TO_ACTION[agent_action.item()]


        
                    elif self.agent_type[policy_id] == 'hipt':
                        z, _, ent, ___, self.npc_lstm_state[policy_id], ____ = self.npc_agent[policy_id].get_z_and_value(obs, self.next_done, self.npc_lstm_state[policy_id])
                        self.z = z
                        agent_action, _, __, ___, _____, ______ = self.npc_agent[policy_id].get_action_and_value(obs, z, self.next_done, self.npc_lstm_state[policy_id])
                        npc_action = Action.INDEX_TO_ACTION[agent_action.item()]
                    
                    else:
                        agent_action,_, _, ___, probs = self.npc_agent[policy_id].get_action_and_value(obs)


                        npc_action = Action.INDEX_TO_ACTION[agent_action.item()]

            else:
                npc_action, _ = policy.action(state)

            # print(npc_action)

            super(OvercookedGame, self).enqueue_action(policy_id, npc_action)

    def is_full(self):
        return self.num_players >= self.max_players

    def is_finished(self):
        # val = not self.layouts and self._curr_game_over()
        val = self._curr_game_over()
        return val

    def is_empty(self):
        """
        Game is considered safe to scrap if there are no active players or if there are no humans (spectating or playing)
        """
        return (
            super(OvercookedGame, self).is_empty()
            or not self.spectators
            and not self.human_players
        )

    def is_ready(self):
        """
        Game is ready to be activated if there are a sufficient number of players and at least one human (spectator or player)
        """
        # info = StreamInfo(name="OvercookedStream", type="Event", channel_count=1, nominal_srate=0, channel_format='string', source_id='Overcooked')
        # self.outlet = StreamOutlet(info)
        # print("Stream outlet created.")
        return super(OvercookedGame, self).is_ready() and not self.is_empty()

    def apply_action(self, player_id, action):
        pass

    def apply_actions(self):
        # Default joint action, as NPC policies and clients probably don't enqueue actions fast
        # enough to produce one at every tick
        joint_action = [Action.STAY] * len(self.players)

        # Synchronize individual player actions into a joint-action as required by overcooked logic
        for i in range(len(self.players)):
            # if this is a human, don't block and inject
            if self.players[i] in self.human_players:
                try:
                    # we don't block here in case humans want to Stay
                    joint_action[i] = self.pending_actions[i].get(block=False)
                except Empty:
                    pass
            else:
                # we block on agent actions to ensure that the agent gets to do one action per state
                joint_action[i] = self.pending_actions[i].get(block=True)

        # Apply overcooked game logic to get state transition
        prev_state = self.state
        self.state, info = self.mdp.get_state_transition(
            prev_state, joint_action
        )
        if self.show_potential:
            self.phi = self.mdp.potential_function(
                prev_state, self.mp, gamma=0.99
            )

        # Send next state to all background consumers if needed
        if self.curr_tick % self.ticks_per_ai_action == 0:
            for npc_id in self.npc_policies:
                self.npc_state_queues[npc_id].put(self.state, block=False)

        # Update score based on soup deliveries that might have occured
        curr_reward = sum(info["sparse_reward_by_agent"])
        self.score += curr_reward
        
        collision = self.mdp.is_prev_step_was_collision 
        if collision:
            self.num_collisions += 1

        # Initialize the stream outlet once
        # if not hasattr(self, 'outlet'):  # Check if outlet has already been created
        #     info = StreamInfo(name="OvercookedStream", type="Event", channel_count=1, nominal_srate=0, channel_format='string',source_id='Overcooked' )
        #     self.outlet = StreamOutlet(info)
        #     print("Stream outlet created.")

        # # Create dummy JSON data
        # dummy_data = {
        #     "event": "dummy_event",
        #     "timestamp": time(),
        #     "value": 42
        # }

            
                
        transition = {
            "state": json.dumps(prev_state.to_dict()),
            "joint_action": json.dumps(joint_action),
            "reward": curr_reward,
            "time_left": max(self.max_time - (time() - self.start_time), 0),
            "score": self.score,
            "time_elapsed": time() - self.start_time,
            "cur_gameloop": self.curr_tick,
            "layout": json.dumps(self.mdp.terrain_mtx),
            "layout_name": self.curr_layout,
            "trial_id": str(self.start_time),
            "player_0_id": self.players[0],
            "player_1_id": self.players[1],
            "player_0_is_human": self.players[0] in self.human_players,
            "player_1_is_human": self.players[1] in self.human_players,
            "collision": collision,
            "num_collisions": self.num_collisions,
            "z": json.dumps(self.z.detach().cpu().numpy().tolist()),
            "unix_timestamp":  time() #datetime.now(timezone.utc).isoformat(timespec='microseconds') 
        }

        # info = StreamInfo(name="OvercookedStream", type="Event", channel_count=1, nominal_srate=0, channel_format='string', source_id='Overcooked')
                
        # # Create the stream outlet
        # outlet = StreamOutlet(info)
        # print("Stream outlet created.")
        
        message = json.dumps(transition)
        # print("Pushing sample:", message)  # Debugging message content
        
        # outlet.push_sample([message])
        # print("Sample pushed.")


        #message = json.dumps(dummy_data)
        # print("Pushing sample:", message)  # Debugging message content
        # self.outlet.push_sample([message]) #=> COMMENTED FOR drl
        # print("Sample pushed.")
        # database1.update_transition(transition, self.commit_hash)
        self.trajectory.append(transition)
        

        # Return about the current transition
        return prev_state, joint_action, info

    def enqueue_action(self, player_id, action):
        overcooked_action = self.action_to_overcooked_action[action]
        super(OvercookedGame, self).enqueue_action(
            player_id, overcooked_action
        )

    def reset(self):
        status = super(OvercookedGame, self).reset()
        if status == self.Status.RESET:
            # Hacky way of making sure game timer doesn't "start" until after reset timeout has passed
            self.start_time += self.reset_timeout / 1000
    def tick(self):
        self.curr_tick += 1
        return super(OvercookedGame, self).tick()
    
    def update_explanation(self, new_adax):
        self.xai_explanation = new_adax
        return super(OvercookedGame, self).update_explanation(new_adax)
    
    def activate(self):
        super(OvercookedGame, self).activate()
        print("=== activating game ===")

        # Sanity check at start of each game
        if not self.npc_players.union(self.human_players) == set(self.players):
            raise ValueError("Inconsistent State")

        self.curr_layout = self.layouts[0]
        self.mdp = OvercookedGridworld.from_layout_name(
            self.curr_layout, **self.mdp_params
        )
        if self.show_potential:
            self.mp = MotionPlanner.from_pickle_or_compute(
                self.mdp, counter_goals=NO_COUNTERS_PARAMS
            )
        self.state = self.mdp.get_standard_start_state()
        if self.show_potential:
            self.phi = self.mdp.potential_function(
                self.state, self.mp, gamma=0.99
            )
        self.start_time = time()
        self.curr_tick = 0
        self.score = 0
        self.threads = []
        for npc_policy in self.npc_policies:
            self.npc_policies[npc_policy].reset()
            self.npc_state_queues[npc_policy].put(self.state)
            t = Thread(target=self.npc_policy_consumer, args=(npc_policy,self.mdp))
            self.threads.append(t)
            t.start()
        print("=== Game activated ===")

    def deactivate(self):
        super(OvercookedGame, self).deactivate()
        print("=== Deactivating game ===")
        # Ensure the background consumers do not hang
        for npc_policy in self.npc_policies:
            self.npc_state_queues[npc_policy].put(self.state)

        print("=== Waiting for all threads to exit ===")
        # Wait for all background threads to exit
        for t in self.threads:
            t.join()

        print("=== All threads exitted. Clearing all action queues ===")
        # Clear all action queues
        self.clear_pending_actions()
        print("=== Deactivated ===")

    def get_state(self):
        state_dict = {}
        state_dict["potential"] = self.phi if self.show_potential else None
        state_dict["state"] = self.state.to_dict()
        state_dict["score"] = self.score
        state_dict["time_left"] = max(
            self.max_time - (time() - self.start_time), 0
        )
        state_dict["current_phase"] = self.current_phase
        state_dict["current_round"] = self.current_round
        state_dict["current_session"] = self.current_session
        state_dict["current_layout"] = self.curr_layout
        state_dict["total_rounds"] = self.total_rounds
        state_dict["layouts"] = self.layouts
        state_dict["xai_explanation"] = self.xai_explanation
        return state_dict
    
    def set_round(self, new_round):
        self.current_round = new_round
    
    def set_session(self, new_session):
        self.current_session = new_session

    def set_phase(self, new_phase):
        self.current_phase = new_phase
        
    def set_layout(self, new_layout):
        self.curr_layout = new_layout

    def to_json(self):
        obj_dict = {}
        obj_dict["terrain"] = self.mdp.terrain_mtx if self._is_active else None
        obj_dict["state"] = self.get_state() if self._is_active else None
        return obj_dict
    def load_bc_model(self,model_dir, verbose=False):
        """
        Returns the model instance (including all compilation data like optimizer state) and a dictionary of parameters
        used to create the model
        """
        if verbose:
            print("Loading bc model from ", model_dir)
        print("Model loading")
        print("Model dir: ", model_dir)
        print("TensorFlow version:", tf.__version__)
        print("Keras version:", tf.keras.__version__)
        model = keras.models.load_model(model_dir, custom_objects={"tf": tf})

        print("Model loaded")
        with open(os.path.join(model_dir, "metadata.pickle"), "rb") as f:
            bc_params = pickle.load(f)
        return model, bc_params
    
    def get_policy(self, npc_id, idx=0):
        print("npc_id: ", npc_id)
        if npc_id.lower().startswith("rllib"):
            print("agent policy", "rllib", npc_id.lower())
            try:
                # Loading rllib agents requires additional helpers
                fpath = os.path.join(AGENT_DIR, npc_id, "agent")
                fix_bc_path(fpath)
                agent = load_agent(fpath, agent_index=idx)
                return agent
            except Exception as e:
                raise IOError(
                    "Error loading Rllib Agent\n{}".format(e.__repr__())
                )
        else:
            try:
                # fpath = os.path.join(AGENT_DIR, npc_id, "agent.pickle")
                # with open(fpath, "rb") as f:
                #     return pickle.load(f)

                #bc_model_path = AGENT_DIR + "/tutorial_notebook_results/BC"
                print("NPC ID: ", npc_id)
                #bc_model_path = os.path.join(AGENT_DIR, npc_id)

                #print("BC model path: ", bc_model_path)
                #bc_model_path = AGENT_DIR + "/PPOCrampedRoom"

                bc_model_path = AGENT_DIR + "/" + npc_id

                bc_model, bc_params = self.load_bc_model(bc_model_path) ##

                bc_policy = BehaviorCloningPolicy.from_model(bc_model, bc_params, stochastic=True)
                base_ae = _get_base_ae(bc_params)
                base_env = base_ae.env

                print("[DEBUG] Layout:", base_env.mdp.layout_name)
                bc_agent0 = RlLibAgent(bc_policy, 0, base_env.featurize_state_mdp)

                bc_agent1 = RlLibAgent(bc_policy, 1, base_env.featurize_state_mdp)
                agent = AgentPair(bc_agent0, bc_agent1)
                print("Agent loaded")
                return  bc_agent1#agent


            except Exception as e:
                raise IOError("Error loading agent\n{}".format(e.__repr__()))
    
    def get_data(self):
        """
        Returns and then clears the accumulated trajectory
        """
        #datetime.now(timezone.utc).isoformat(timespec='microseconds'),
        print("UID is:", self.get_uid())
        print("Current round Id: ", self.current_round)
        data = {
            "uid": self.get_uid(),#str(time()),
            "trajectory": self.trajectory,
            "unix_timestamp":  time(), #datetime.now(timezone.utc).isoformat(timespec='microseconds'),# str(time.time()),
            "round_id": self.id,
            "round_hash": str(generate_unique_hash())
        }
        self.trajectory = []
        # if we want to store the data and there is data to store
        if self.write_data and len(data["trajectory"]) > 0:
            # configs = self.write_config
            # create necessary dirs
            # data_path = create_dirs(configs, self.curr_layout)
            # the 3-layer-directory structure should be able to uniquely define any experiment
            # with open(os.path.join(data_path, "result.pkl"), "wb") as f:
            #     pickle.dump(data, f)
                # 1 single table - timestamp, uid, string of self.trajectory
                # 1.5 single table - timestamp, uid, expanded self.trajectory
                # 2 double table - timestamp, uid, foreign key :: foreign key, other fields of self.trajectory
            # insert the database table update logic
            database.update(data) 
            # database1.update(data)
            
            

        # self.stop_tracking()
        
        return data


class OvercookedTutorial(OvercookedGame):

    """
    Wrapper on OvercookedGame that includes additional data for tutorial mechanics, most notably the introduction of tutorial "phases"

    Instance Variables:
        - curr_phase (int): Indicates what tutorial phase we are currently on
        - phase_two_score (float): The exact sparse reward the user must obtain to advance past phase 2
    """

    def __init__(
        self,
        layouts=["tutorial_0"],
        mdp_params={},
        playerZero="human",
        playerOne="AI",
        phaseTwoScore=63,
        **kwargs
    ):
        super(OvercookedTutorial, self).__init__(
            layouts=layouts,
            mdp_params=mdp_params,
            playerZero=playerZero,
            playerOne=playerOne,
            showPotential=False,
            **kwargs
        )
        self.phase_two_score = phaseTwoScore
        self.phase_two_finished = False
        self.max_time = 0
        self.max_players = 2
        self.ticks_per_ai_action = 1
        self.curr_phase = 0
        # we don't collect tutorial data
        self.write_data = False

    @property
    def reset_timeout(self):
        return 1

    def needs_reset(self):
        if self.curr_phase <= len(self.layouts):
            return self.score > 0
        return False
    
    def is_finished(self):
        return self.curr_phase > 3 and self.score >= 0 # changed to 0 from  float("inf") TODO: game ends early fix

    def reset(self):
        super(OvercookedTutorial, self).reset()
        self.curr_phase += 1

    def get_policy(self, *args, **kwargs):
        return TutorialAI()

    def apply_actions(self):
        """
        Apply regular MDP logic with retroactive score adjustment tutorial purposes
        """
        _, _, info = super(OvercookedTutorial, self).apply_actions()

        human_reward, ai_reward = info["sparse_reward_by_agent"]
        # We only want to keep track of the human's score in the tutorial
        self.score -= ai_reward
        # Phase two requires a specific reward to complete
        # if self.curr_phase == 2:
        #     self.score = 0
            # if human_reward == self.phase_two_score:
            #     self.phase_two_finished = True


class DummyOvercookedGame(OvercookedGame):
    """
    Class that hardcodes the AI to be random. Used for debugging
    """

    def __init__(self, layouts=["cramped_room"], **kwargs):
        super(DummyOvercookedGame, self).__init__(layouts, **kwargs)

    def get_policy(self, *args, **kwargs):
        return DummyAI()


class DummyAI:
    """
    Randomly samples actions. Used for debugging
    """

    def action(self, state):
        [action] = random.sample(
            [
                Action.STAY,
                Direction.NORTH,
                Direction.SOUTH,
                Direction.WEST,
                Direction.EAST,
                Action.INTERACT,
            ],
            1,
        )
        return action, None

    def reset(self):
        pass


class DummyComputeAI(DummyAI):
    """
    Performs simulated compute before randomly sampling actions. Used for debugging
    """

    def __init__(self, compute_unit_iters=1e5):
        """
        compute_unit_iters (int): Number of for loop cycles in one "unit" of compute. Number of
                                    units performed each time is randomly sampled
        """
        super(DummyComputeAI, self).__init__()
        self.compute_unit_iters = int(compute_unit_iters)

    def action(self, state):
        # Randomly sample amount of time to busy wait
        iters = random.randint(1, 10) * self.compute_unit_iters

        # Actually compute something (can't sleep) to avoid scheduling optimizations
        val = 0
        for i in range(iters):
            # Avoid branch prediction optimizations
            if i % 2 == 0:
                val += 1
            else:
                val += 2

        # Return randomly sampled action
        return super(DummyComputeAI, self).action(state)


class StayAI:
    """
    Always returns "stay" action. Used for debugging
    """

    def action(self, state):
        return Action.STAY, None

    def reset(self):
        pass


class TutorialAI:
    COOK_SOUP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Grab second onion
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Grab third onion
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Cook soup
        Action.INTERACT,
        # Grab plate
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,
        Direction.WEST,
        Direction.NORTH,
        # Deliver soup
        Action.INTERACT,
        Direction.EAST,
        Direction.EAST,
        Direction.EAST,
        Action.INTERACT,
        Direction.WEST,
    ]

    COOK_SOUP_COOP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,
        # Move to start so this loops
        Direction.EAST,
        Direction.EAST,
        # Pause to make cooperation more real time
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
    ]

    def __init__(self):
        self.curr_phase = -1
        self.curr_tick = -1

    def action(self, state):
        self.curr_tick += 1
        if self.curr_phase <= 4:
            return (
                self.COOK_SOUP_LOOP[self.curr_tick % len(self.COOK_SOUP_LOOP)],
                None,
            )
        # elif self.curr_phase == 2:
        #     return (
        #         self.COOK_SOUP_COOP_LOOP[
        #             self.curr_tick % len(self.COOK_SOUP_COOP_LOOP)
        #         ],
        #         None,
        #     )
        return Action.STAY, None

    def reset(self):
        self.curr_tick = -1
        self.curr_phase += 1
