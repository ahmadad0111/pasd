import os
import gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from trainers.hipt_trainer import HiPTTrainer
from trainers.diayn_trainer import DIAYNTrainer
from trainers.hipt_dynamic_trainer import HiPTTrainer_dyn
from utils.utils import make_env
from tqdm import tqdm
from trainers.riayn_trainer import RIAYNTrainer

layout_name_map = {
        "cramped_room": "Cramped Room",
        "asymmetric_advantages": "Asymmetric Advantages",
        "counter_circuit_o_1order": "Counter Circuit",
        "forced_coordination": "Forced Coordination",
        "coordination_ring": "Coordination Ring",
    }

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif',fps = 60):

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=fps)
    anim.save(path + filename, writer='imagemagick', fps=fps)

    plt.close()

def eval_rollout(trainer, config):
    envs = gym.vector.SyncVectorEnv(
        [make_env(config.gym_id, config.layout.name, config.seed + config.training.num_envs + i) for i in range(config.training.population_size*2)]
    )
    initial_obs = envs.reset()

    cum_rew = np.zeros(config.training.population_size*2)
    if isinstance(trainer, HiPTTrainer) or isinstance(trainer, HiPTTrainer_dyn):
        curr_p_len = np.random.randint(config.training.p_range[0], config.training.p_range[1], size = config.training.population_size*2)

    next_obs_p0 = torch.Tensor(initial_obs[:,0]).to(trainer.device)
    next_obs_p1 = torch.Tensor(initial_obs[:,1]).to(trainer.device)

    lstm_state = (
        torch.zeros(trainer.agent.lstm.num_layers, config.training.population_size*2, trainer.agent.lstm.hidden_size).to(trainer.device),
        torch.zeros(trainer.agent.lstm.num_layers, config.training.population_size*2, trainer.agent.lstm.hidden_size).to(trainer.device),
    )   

    if config.evaluation.visualize_ep:
        visualizer = StateVisualizer()
        vis_path = os.path.join(os.getcwd(), "visualizations")
        if not os.path.exists(vis_path):
            os.mkdir(vis_path)
        ep_visualizations = [[] for _ in range(config.training.population_size*2)]

    next_done = torch.zeros(2*config.training.population_size).to(trainer.device)

    print("Evaluating agent with SP Population.....")
    state_embeddings = []
    all_z_step = []
    entropy_z = []

    for step in range(config.training.rollout_steps):
        current_lstm_state = (lstm_state[0].clone(), lstm_state[1].clone())
        agent_obs = torch.cat((next_obs_p0[:config.training.population_size], next_obs_p1[config.training.population_size:]), dim = 0)
        
        if config.evaluation.visualize_ep:
            for ind,env in enumerate(envs.envs):
                hud_data = {"Timestep" : step, "Reward" : cum_rew[ind]}
                surface = visualizer.render_state(env.base_env.state, grid = env.mdp.terrain_mtx, hud_data = hud_data)
                state_imgdata = pygame.surfarray.array3d(surface).swapaxes(0,1)
                ep_visualizations[ind].append(state_imgdata)

        if isinstance(trainer, HiPTTrainer) or isinstance(trainer, HiPTTrainer_dyn):
            z, _, ent, ___, lstm_state, ____ = trainer.agent.get_z_and_value(agent_obs, next_done, lstm_state)

            ######### collect state features and save them
            state_embeddings.append(trainer.agent.get_hifeat(agent_obs, next_done, lstm_state))

            if step == 0:
                current_z = z.clone()
            else:
                for p_step in range(config.training.population_size*2):
                    if curr_p_len[p_step] == 0:
                        curr_p_len[p_step] = np.random.randint(config.training.p_range[0], config.training.p_range[1])
                        current_z[p_step] = z[p_step]
            agent_actions, _, __, ___, _____, ______ = trainer.agent.get_action_and_value(agent_obs, current_z, next_done, current_lstm_state)

            all_z_step.append(current_z.clone())
            entropy_z.append(ent.clone())

        else:
            agent_actions, _, __, ___, lstm_state = trainer.agent.get_action_and_value(agent_obs, next_done, current_lstm_state)

        partner_actions = torch.zeros(config.training.population_size*2).to(trainer.device)
        for ind, partner in enumerate(trainer.train_partners):
            partner_obs = torch.cat((next_obs_p1[ind:ind+1], next_obs_p0[ind+config.training.population_size:ind+config.training.population_size + 1]), dim = 0)
            partner_action,__, _, ___, ____ = partner.get_action_and_value(partner_obs)
            partner_actions[ind] = partner_action[0]
            partner_actions[ind+config.training.population_size] = partner_action[1]    

        actions_p0 = torch.cat((agent_actions[:config.training.population_size], partner_actions[:config.training.population_size]), dim = 0)
        actions_p1 = torch.cat((partner_actions[:config.training.population_size], agent_actions[:config.training.population_size]), dim = 0)
        joint_action = torch.cat((actions_p0.view(2*config.training.population_size,1),actions_p1.view(2*config.training.population_size,1)),1)
        joint_action = joint_action.type(torch.int8)
        next_obs, reward, done, info = envs.step(joint_action.cpu().numpy())

        next_obs_p0 = torch.Tensor(next_obs[:,0]).to(trainer.device)
        next_obs_p1 = torch.Tensor(next_obs[:,1]).to(trainer.device)
        print(f"Observation shape : {next_obs_p0.shape}")
        print(f"Observation shape : {next_obs_p1.shape}")
        next_done = torch.Tensor(done).to(trainer.device)

        cum_rew += reward

        if isinstance(trainer, HiPTTrainer) or isinstance(trainer, HiPTTrainer_dyn):
            curr_p_len -= 1
    
    envs.close()
    if isinstance(trainer, HiPTTrainer) or isinstance(trainer, HiPTTrainer_dyn):
        state_embeddings_tensor = torch.stack(state_embeddings)  
        torch.save(state_embeddings_tensor, "state_embeddings.pt")
        all_z_step_tensor = torch.stack(all_z_step)
        torch.save(all_z_step_tensor, "all_z_step.pt")
        print("Saved to", os.path.abspath("state_embeddings.pt"))
        all_ent_step_tensor = torch.stack(entropy_z)
        torch.save(all_ent_step_tensor, "all_ent_step.pt")

    total_rew = 0
    total_rew_blue = 0
    total_rew_green = 0
    count_B = 0
    count_G = 0
    for ind, reward in enumerate(cum_rew):
        total_rew += reward
        if ind < config.training.population_size:
            print(f"Blue : {config.model.name} Agent, Green : SP Agent {ind}, Total reward: {reward}")
            total_rew_blue += reward
            count_B += 1
            if config.evaluation.visualize_ep:
                save_frames_as_gif(ep_visualizations[ind], vis_path, f"/{config.layout.name}_blue_{config.model.name}_green_spagent{ind}.gif")   
        else:
            print(f"Blue : SP Agent {ind%config.training.population_size}, Green : {config.model.name} Agent, Total reward: {reward}")
            total_rew_green += reward
            count_G += 1
            if config.evaluation.visualize_ep:
                save_frames_as_gif(ep_visualizations[ind], vis_path, f"/{config.layout.name}_blue_spagent{ind%config.training.population_size}_green_{config.model.name}.gif")
    
    print(f"Total mean reward: {total_rew / len(cum_rew)} || Total reward as Blue agent: {total_rew_blue / count_B} || Total reward as Green agent: {total_rew_green / count_G}")



def plot_sp_heatmap(trainer, config,path):
    envs = gym.vector.SyncVectorEnv(
        [make_env(config.gym_id, config.layout.name, config.seed + config.training.num_envs + i) for i in range(config.training.population_size)]
    )
    initial_obs = envs.reset()

    if config.layout.eval_partner_pop_path is not None:
        trainer.load_population(config.evaluation.agent_type,config.layout.eval_partner_pop_path)
    
    trainer.load_population(config.evaluation.agent_type,path)

    #path = "sp_populoation/cramped_room"

    vis_path = os.path.join(os.getcwd(), "visualizations")
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)

    rew_mat = np.zeros((config.training.population_size, config.training.population_size))

    headers = [f"Agent {i}" for i in range(config.training.population_size)]

    rows = [[f"Agent {i}"] for i in range(config.training.population_size)]

    next_obs_p0 = torch.Tensor(initial_obs[:,0]).to(trainer.device)
    next_obs_p1 = torch.Tensor(initial_obs[:,1]).to(trainer.device)

    for agent_ind in range(config.training.population_size):
        print(f"Running rollouts for Agent {agent_ind}.....")
        for step in tqdm(range(config.training.rollout_steps)):
            agent_actions,__, _, ___, ____ = trainer.agent_pop[agent_ind].get_action_and_value(next_obs_p0)
            partner_actions = torch.zeros(config.training.population_size).to(trainer.device)
            for partner_ind in range(config.training.population_size):                 
                partner_action,__, _, ___, ____ = trainer.agent_pop[partner_ind].get_action_and_value(next_obs_p1[partner_ind:partner_ind+1])
                partner_actions[partner_ind] = partner_action[0]

            joint_action = torch.cat((agent_actions.view(config.training.population_size,1),partner_actions.view(config.training.population_size,1)),1)
            joint_action = joint_action.type(torch.int8)
            next_obs, reward, done, info = envs.step(joint_action.cpu().numpy())

            rew_mat[agent_ind] += reward

            next_obs_p0 = torch.Tensor(next_obs[:,0]).to(trainer.device)
            next_obs_p1 = torch.Tensor(next_obs[:,1]).to(trainer.device)
        rows[agent_ind].extend([str(rew) for rew in rew_mat[agent_ind]])
    
    # print("Plotting Heatmap.....")

    # plt.rcParams["figure.figsize"] = (20,15)
    # labels = [rows[i][1:] for i in range(config.training.population_size)]
    
    # ax = sns.heatmap(rew_mat, cmap="inferno", annot=labels, annot_kws={'fontsize': 8}, fmt='s', xticklabels = headers, yticklabels = headers)
    # ax.set(xlabel="", ylabel="")
    # ax.xaxis.tick_top()
    # plt.title(layout_name_map[f"{config.layout.name}"])
    # plt.savefig(vis_path + f"/sp_{config.layout.name}_heatmap_.png",bbox_inches='tight')
    # plt.close()

    # print("Done!")
    print("Plotting Heatmap.....")

    plt.rcParams["figure.figsize"] = (20, 15)
    labels = [rows[i][1:] for i in range(config.training.population_size)]

    # Enhanced heatmap
    ax = sns.heatmap(
        rew_mat,
        cmap="magma",               # Better perceptual colormap
        annot=labels,
        annot_kws={'fontsize': 8, 'rotation': 45},  # Rotate annotations slightly
        fmt='s',
        xticklabels=headers,
        yticklabels=headers,
        linewidths=0.5,             # Add grid lines
        linecolor='gray',
        square=True,                # Make cells square
        cbar_kws={'label': 'Reward'}  # Colorbar label
    )

    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()             # Put x-axis on top
    plt.title(layout_name_map[f"{config.layout.name}"], fontsize=18)
    plt.tight_layout()

    plt.savefig(vis_path + f"/sp_{config.layout.name}_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()

    print("Done!")

    
def eval_rollout_dyn(trainer, config):

    envs = gym.vector.SyncVectorEnv(
        [make_env(config.gym_id, config.layout.name, config.seed + config.training.num_envs + i) for i in range(config.training.population_size*2)]
    )
    initial_obs = envs.reset()
    expanded_z = torch.tile(torch.arange(config.layout.z_dim).view(-1,1),(config.training.num_envs,)).reshape(-1).to(trainer.device)
    cum_rew = np.zeros(config.training.population_size*2)
    # if isinstance(trainer, HiPTTrainer_dyn):
    #     curr_p_len = np.random.randint(config.training.p_range[0], config.training.p_range[1], size = config.training.population_size*2)

    next_obs_p0 = torch.Tensor(initial_obs[:,0]).to(trainer.device)
    next_obs_p1 = torch.Tensor(initial_obs[:,1]).to(trainer.device)

    lstm_state = (
        torch.zeros(trainer.agent.lstm.num_layers, config.training.population_size*2, trainer.agent.lstm.hidden_size).to(trainer.device),
        torch.zeros(trainer.agent.lstm.num_layers, config.training.population_size*2, trainer.agent.lstm.hidden_size).to(trainer.device),
    )   

    if config.evaluation.visualize_ep:
        visualizer = StateVisualizer()
        vis_path = os.path.join(os.getcwd(), "visualizations")
        if not os.path.exists(vis_path):
            os.mkdir(vis_path)
        ep_visualizations = [[] for _ in range(config.training.population_size*2)]

    next_done = torch.zeros(2*config.training.population_size).to(trainer.device)

    print("Evaluating agent with SP Population.....")

    state_embeddings = []
    all_z_step = []

    for step in range(config.training.rollout_steps):
        current_lstm_state = (lstm_state[0].clone(), lstm_state[1].clone())
        agent_obs = torch.cat((next_obs_p0[:config.training.population_size], next_obs_p1[config.training.population_size:]), dim = 0)

         
        
        
        if config.evaluation.visualize_ep:
            for ind,env in enumerate(envs.envs):
                hud_data = {"Timestep" : step, "Reward" : cum_rew[ind]}
                surface = visualizer.render_state(env.base_env.state, grid = env.mdp.terrain_mtx, hud_data = hud_data)
                state_imgdata = pygame.surfarray.array3d(surface).swapaxes(0,1)
                ep_visualizations[ind].append(state_imgdata)

        if isinstance(trainer, HiPTTrainer_dyn):
            z, _, __, ___, lstm_state, ____ = trainer.agent.get_z_and_value(agent_obs, next_done, lstm_state)

            ######### collect state features and save them
            state_embeddings.append(trainer.agent.get_hifeat(agent_obs, next_done, lstm_state))
            
            # term_action_full = term_action_full.reshape(config.layout.z_dim, config.training.num_envs)
            

            if step == 0:
                current_z = z.clone()
            else:
                # term_action = torch.gather(term_action_full, 0,current_z.view(1,-1)).reshape(-1)
                term_action, _, _, _, _, _ = trainer.agent.get_termination_and_value(agent_obs, current_z, next_done, current_lstm_state)
                for p_step in range(config.training.population_size*2):
                    #if curr_p_len[p_step] == 0:
                    if term_action[p_step].item() == 1:
                        #curr_p_len[p_step] = np.random.randint(config.training.p_range[0], config.training.p_range[1])
                        current_z[p_step] = z[p_step]
            agent_actions, _, __, ___, _____, ______ = trainer.agent.get_action_and_value(agent_obs, current_z, next_done, current_lstm_state)

            all_z_step.append(z.clone())

        else:
            agent_actions, _, __, ___, lstm_state = trainer.agent.get_action_and_value(agent_obs, next_done, current_lstm_state)

        partner_actions = torch.zeros(config.training.population_size*2).to(trainer.device)
        for ind, partner in enumerate(trainer.train_partners):
            partner_obs = torch.cat((next_obs_p1[ind:ind+1], next_obs_p0[ind+config.training.population_size:ind+config.training.population_size + 1]), dim = 0)
            partner_action,__, _, ___, ____ = partner.get_action_and_value(partner_obs)
            partner_actions[ind] = partner_action[0]
            partner_actions[ind+config.training.population_size] = partner_action[1]    

        actions_p0 = torch.cat((agent_actions[:config.training.population_size], partner_actions[:config.training.population_size]), dim = 0)
        actions_p1 = torch.cat((partner_actions[:config.training.population_size], agent_actions[:config.training.population_size]), dim = 0)
        joint_action = torch.cat((actions_p0.view(2*config.training.population_size,1),actions_p1.view(2*config.training.population_size,1)),1)
        joint_action = joint_action.type(torch.int8)
        next_obs, reward, done, info = envs.step(joint_action.cpu().numpy())

        next_obs_p0 = torch.Tensor(next_obs[:,0]).to(trainer.device)
        next_obs_p1 = torch.Tensor(next_obs[:,1]).to(trainer.device)
        next_done = torch.Tensor(done).to(trainer.device)

        cum_rew += reward

        # if isinstance(trainer, HiPTTrainer) or isinstance(trainer, HiPTTrainer_dyn):
        #     curr_p_len -= 1
    
    envs.close()

    state_embeddings_tensor = torch.stack(state_embeddings)  
    torch.save(state_embeddings_tensor, "state_embeddings.pt")
    print("Saved to", os.path.abspath("state_embeddings.pt"))
    all_z_step_tensor = torch.stack(all_z_step)
    torch.save(all_z_step_tensor, "all_z_step.pt")


    total_rew = 0
    total_rew_blue = 0
    total_rew_green = 0
    count_B = 0
    count_G = 0
    for ind, reward in enumerate(cum_rew):
        total_rew += reward
        if ind < config.training.population_size:
            print(f"Blue : {config.model.name} Agent, Green : SP Agent {ind}, Total reward: {reward}")
            total_rew_blue += reward
            count_B += 1
            if config.evaluation.visualize_ep:
                save_frames_as_gif(ep_visualizations[ind], vis_path, f"/{config.layout.name}_blue_{config.model.name}_green_spagent{ind}.gif")   
        else:
            print(f"Blue : SP Agent {ind%config.training.population_size}, Green : {config.model.name} Agent, Total reward: {reward}")
            total_rew_green += reward
            count_G += 1
            if config.evaluation.visualize_ep:
                save_frames_as_gif(ep_visualizations[ind], vis_path, f"/{config.layout.name}_blue_spagent{ind%config.training.population_size}_green_{config.model.name}.gif")
    
    print(f"Total mean reward: {total_rew / len(cum_rew)} || Total reward as Blue agent: {total_rew_blue / count_B} || Total reward as Green agent: {total_rew_green / count_G}")

def eval_rollout_diayn(trainer, config):
    envs = gym.vector.SyncVectorEnv(
        [make_env(config.gym_id, config.layout.name, config.seed + config.training.num_envs + i) for i in range(config.training.population_size*2)]
    )
    initial_obs = envs.reset()

    cum_rew = np.zeros(config.training.population_size*2)
    if isinstance(trainer, HiPTTrainer) or isinstance(trainer, HiPTTrainer_dyn) or isinstance(trainer, RIAYNTrainer) or isinstance(trainer, DIAYNTrainer) :
        curr_p_len = np.random.randint(config.training.p_range[0], config.training.p_range[1], size = config.training.population_size*2)

    next_obs_p0 = torch.Tensor(initial_obs[:,0]).to(trainer.device)
    next_obs_p1 = torch.Tensor(initial_obs[:,1]).to(trainer.device)

    lstm_state_lo = (
        torch.zeros(trainer.lo_agent.lstm.num_layers, config.training.population_size*2, trainer.lo_agent.lstm.hidden_size).to(trainer.device),
        torch.zeros(trainer.lo_agent.lstm.num_layers, config.training.population_size*2, trainer.lo_agent.lstm.hidden_size).to(trainer.device),
    )   
    lstm_state_hi = (
        torch.zeros(trainer.hi_agent.lstm.num_layers, config.training.population_size*2, trainer.hi_agent.lstm.hidden_size).to(trainer.device),
        torch.zeros(trainer.hi_agent.lstm.num_layers, config.training.population_size*2, trainer.hi_agent.lstm.hidden_size).to(trainer.device),
    )   
    lstm_state_disc = (
        torch.zeros(trainer.discriminator.lstm.num_layers, config.training.population_size*2, trainer.discriminator.lstm.hidden_size).to(trainer.device),
        torch.zeros(trainer.discriminator.lstm.num_layers, config.training.population_size*2, trainer.discriminator.lstm.hidden_size).to(trainer.device),
    )  
    if config.evaluation.visualize_ep:
        visualizer = StateVisualizer()
        vis_path = os.path.join(os.getcwd(), "visualizations")
        if not os.path.exists(vis_path):
            os.mkdir(vis_path)
        ep_visualizations = [[] for _ in range(config.training.population_size*2)]

    next_done = torch.zeros(2*config.training.population_size).to(trainer.device)

    print("Evaluating agent with SP Population.....")
    state_embeddings = []
    all_z_step = []
    entropy_z = []
    all_preds_step = []

    obs_agent0 = []
    obs_agent1 = []

    for step in range(config.training.rollout_steps):

        z_value = (step // 20) % config.layout.z_dim
        current_z = torch.full(
            (config.training.population_size*2,),
            z_value,
            dtype=torch.int64,
            device=trainer.device
        )



        current_lstm_state_lo = (lstm_state_lo[0].clone(), lstm_state_lo[1].clone())
        current_lstm_state_disc = (lstm_state_disc[0].clone(), lstm_state_disc[1].clone())
        current_lstm_state_hi = (lstm_state_hi[0].clone(), lstm_state_hi[1].clone())
        
        agent_obs = torch.cat((next_obs_p0[:config.training.population_size], next_obs_p1[config.training.population_size:]), dim = 0)
        
        if config.evaluation.visualize_ep:
            for ind,env in enumerate(envs.envs):
                hud_data = {"Timestep" : step, "Reward" : cum_rew[ind]}
                surface = visualizer.render_state(env.base_env.state, grid = env.mdp.terrain_mtx, hud_data = hud_data)
                state_imgdata = pygame.surfarray.array3d(surface).swapaxes(0,1)
                ep_visualizations[ind].append(state_imgdata)

        if isinstance(trainer, DIAYNTrainer) or isinstance(trainer, RIAYNTrainer):
            z, _, ent, ___, lstm_state_hi, ____ = trainer.hi_agent.get_z_and_value(agent_obs, next_done, lstm_state_hi)

            # ######### collect state features and save them
            state_embeddings.append(trainer.hi_agent.get_high_feat(agent_obs, next_done, lstm_state_hi))

            if step == 0:
                current_z = z.clone()
            else:
                for p_step in range(config.training.population_size*2):
                    if curr_p_len[p_step] == 0:
                        curr_p_len[p_step] = np.random.randint(config.training.p_range[0], config.training.p_range[1])
                        current_z[p_step] = z[p_step]


            agent_actions, _, __, ___, _____, ______, lstm_state_lo = trainer.lo_agent.get_action_and_value(agent_obs, current_z, next_done, current_lstm_state_lo)

            disc_logits, lstm_state_disc = trainer.discriminator.get_disc_logits(agent_obs, next_done, current_lstm_state_disc)

            preds = disc_logits.argmax(dim=-1)

            

            all_z_step.append(z.clone())
            entropy_z.append(ent.clone())
            # all_preds_step.append(current_z.clone())

        partner_actions = torch.zeros(config.training.population_size*2).to(trainer.device)
        for ind, partner in enumerate(trainer.train_partners):
            partner_obs = torch.cat((next_obs_p1[ind:ind+1], next_obs_p0[ind+config.training.population_size:ind+config.training.population_size + 1]), dim = 0)
            partner_action,__, _, ___, ____ = partner.get_action_and_value(partner_obs)
            partner_actions[ind] = partner_action[0]
            partner_actions[ind+config.training.population_size] = partner_action[1]    

        actions_p0 = torch.cat((agent_actions[:config.training.population_size], partner_actions[:config.training.population_size]), dim = 0)
        actions_p1 = torch.cat((partner_actions[:config.training.population_size], agent_actions[:config.training.population_size]), dim = 0)
        joint_action = torch.cat((actions_p0.view(2*config.training.population_size,1),actions_p1.view(2*config.training.population_size,1)),1)
        joint_action = joint_action.type(torch.int8)
        next_obs, reward, done, info = envs.step(joint_action.cpu().numpy())

        obs_agent0.append(next_obs_p0)
        obs_agent1.append(next_obs_p1)
        

        next_obs_p0 = torch.Tensor(next_obs[:,0]).to(trainer.device)
        next_obs_p1 = torch.Tensor(next_obs[:,1]).to(trainer.device)
        next_done = torch.Tensor(done).to(trainer.device)

        cum_rew += reward

        if isinstance(trainer, HiPTTrainer) or isinstance(trainer, DIAYNTrainer) or isinstance(trainer, RIAYNTrainer):
            curr_p_len -= 1
    
    envs.close()
    state_embeddings_tensor = torch.stack(state_embeddings)  
    torch.save(state_embeddings_tensor, "state_embeddings.pt")
    all_z_step_tensor = torch.stack(all_z_step)
    torch.save(all_z_step_tensor, "all_z_step.pt")

    torch.save(torch.stack(obs_agent0) , "all_obs_agent0")
    torch.save(torch.stack(obs_agent0) , "all_obs_agent1")
    all_ent_step_tensor = torch.stack(entropy_z)
    torch.save(all_ent_step_tensor, "all_ent_step.pt")

    # all_preds_step_tensor = torch.stack(all_preds_step)
    # torch.save(all_preds_step_tensor, "all_disc_preds_step.pt")

    # print("Saved to", os.path.abspath("state_embeddings.pt"))

    # save_frames_as_gif(ep_visualizations[0], vis_path, f"/{config.layout.name}_blue_{config.model.name}_green_spagent{0}.gif")
    # print(f"Current z : {all_z_step_tensor[:,0]}")
    # print(f"Preds z : {all_preds_step_tensor[:,0]}")
     

    total_rew = 0
    total_rew_blue = 0
    total_rew_green = 0
    count_B = 0
    count_G = 0
    for ind, reward in enumerate(cum_rew):
        total_rew += reward
        if ind < config.training.population_size:
            print(f"Blue : {config.model.name} Agent, Green : SP Agent {ind}, Total reward: {reward}")
            total_rew_blue += reward
            count_B += 1
            if config.evaluation.visualize_ep:
                save_frames_as_gif(ep_visualizations[ind], vis_path, f"/{config.layout.name}_blue_{config.model.name}_green_spagent{ind}.gif")   
        else:
            print(f"Blue : SP Agent {ind%config.training.population_size}, Green : {config.model.name} Agent, Total reward: {reward}")
            total_rew_green += reward
            count_G += 1
            if config.evaluation.visualize_ep:
                save_frames_as_gif(ep_visualizations[ind], vis_path, f"/{config.layout.name}_blue_spagent{ind%config.training.population_size}_green_{config.model.name}.gif")
    
    print(f"Total mean reward: {total_rew / len(cum_rew)} || Total reward as Blue agent: {total_rew_blue / count_B} || Total reward as Green agent: {total_rew_green / count_G}")




def eval_rollout_diayn1(trainer, config):
    envs = gym.vector.SyncVectorEnv(
        [make_env(config.gym_id, config.layout.name, config.seed + config.training.num_envs + i) for i in range(config.training.population_size*2)]
    )
    initial_obs = envs.reset()

    cum_rew = np.zeros(config.training.population_size*2)
    if isinstance(trainer, HiPTTrainer) or isinstance(trainer, HiPTTrainer_dyn) or isinstance(trainer, RIAYNTrainer) or isinstance(trainer, DIAYNTrainer) :
        curr_p_len = np.random.randint(config.training.p_range[0], config.training.p_range[1], size = config.training.population_size*2)

    next_obs_p0 = torch.Tensor(initial_obs[:,0]).to(trainer.device)
    next_obs_p1 = torch.Tensor(initial_obs[:,1]).to(trainer.device)

    lstm_state_lo = (
        torch.zeros(trainer.lo_agent.lstm.num_layers, config.training.population_size*2, trainer.lo_agent.lstm.hidden_size).to(trainer.device),
        torch.zeros(trainer.lo_agent.lstm.num_layers, config.training.population_size*2, trainer.lo_agent.lstm.hidden_size).to(trainer.device),
    )   
    lstm_state_hi = (
        torch.zeros(trainer.hi_agent.lstm.num_layers, config.training.population_size*2, trainer.hi_agent.lstm.hidden_size).to(trainer.device),
        torch.zeros(trainer.hi_agent.lstm.num_layers, config.training.population_size*2, trainer.hi_agent.lstm.hidden_size).to(trainer.device),
    )   
    lstm_state_disc = (
        torch.zeros(trainer.discriminator.lstm.num_layers, config.training.population_size*2, trainer.discriminator.lstm.hidden_size).to(trainer.device),
        torch.zeros(trainer.discriminator.lstm.num_layers, config.training.population_size*2, trainer.discriminator.lstm.hidden_size).to(trainer.device),
    )  
    if config.evaluation.visualize_ep:
        visualizer = StateVisualizer()
        vis_path = os.path.join(os.getcwd(), "visualizations")
        if not os.path.exists(vis_path):
            os.mkdir(vis_path)
        ep_visualizations = [[] for _ in range(config.training.population_size*2)]

    next_done = torch.zeros(2*config.training.population_size).to(trainer.device)

    print("Evaluating agent with SP Population.....")
    state_embeddings = []
    all_z_step = []
    all_preds_step = []
    entropy_z = []

    obs_agent0 = []
    obs_agent1 = []

    for step in range(config.training.rollout_steps):


        # Choose a fixed skill index, e.g., 0, 1, 2, or 3
        fixed_z_index = 5  # change this to whichever skill you want

        # Create a tensor filled with that skill for all agents in the batch
        current_z = torch.full(
            (config.training.population_size * 2,),  # batch size for your environment
            fixed_z_index,
            dtype=torch.int64,
            device=trainer.device
        )



        current_lstm_state_lo = (lstm_state_lo[0].clone(), lstm_state_lo[1].clone())
        current_lstm_state_disc = (lstm_state_disc[0].clone(), lstm_state_disc[1].clone())
        current_lstm_state_hi = (lstm_state_hi[0].clone(), lstm_state_hi[1].clone())
        
        agent_obs = torch.cat((next_obs_p0[:config.training.population_size], next_obs_p1[config.training.population_size:]), dim = 0)
        
        if config.evaluation.visualize_ep:
            for ind,env in enumerate(envs.envs):
                hud_data = {"Timestep" : step, "Reward" : cum_rew[ind]}
                surface = visualizer.render_state(env.base_env.state, grid = env.mdp.terrain_mtx, hud_data = hud_data)
                state_imgdata = pygame.surfarray.array3d(surface).swapaxes(0,1)
                ep_visualizations[ind].append(state_imgdata)

        if isinstance(trainer, DIAYNTrainer) or isinstance(trainer, RIAYNTrainer):
            z, _, ent, ___, lstm_state_hi, ____ = trainer.hi_agent.get_z_and_value(agent_obs, next_done, lstm_state_hi)

            # ######### collect state features and save them
            state_embeddings.append(trainer.hi_agent.get_high_feat(agent_obs, next_done, lstm_state_hi))

            if step == 0:
                current_z = current_z
                # entropy_ = ent
            else:
                for p_step in range(config.training.population_size*2):
                    if curr_p_len[p_step] == 0:
                        curr_p_len[p_step] = np.random.randint(config.training.p_range[0], config.training.p_range[1])
                        current_z[p_step] = current_z[p_step]
                        # entropy_ = ent


            agent_actions, _, __, ___, _____, ______, lstm_state_lo = trainer.lo_agent.get_action_and_value(agent_obs, current_z, next_done, current_lstm_state_lo)

            disc_logits, lstm_state_disc = trainer.discriminator.get_disc_logits(agent_obs, next_done, current_lstm_state_disc)

            preds = disc_logits.argmax(dim=-1)

            

            all_z_step.append(z.clone())
            entropy_z.append(ent.clone())
            # all_preds_step.append(current_z.clone())

        partner_actions = torch.zeros(config.training.population_size*2).to(trainer.device)
        for ind, partner in enumerate(trainer.train_partners):
            partner_obs = torch.cat((next_obs_p1[ind:ind+1], next_obs_p0[ind+config.training.population_size:ind+config.training.population_size + 1]), dim = 0)
            partner_action,__, _, ___, ____ = partner.get_action_and_value(partner_obs)
            partner_actions[ind] = partner_action[0]
            partner_actions[ind+config.training.population_size] = partner_action[1]    

        actions_p0 = torch.cat((agent_actions[:config.training.population_size], partner_actions[:config.training.population_size]), dim = 0)
        actions_p1 = torch.cat((partner_actions[:config.training.population_size], agent_actions[:config.training.population_size]), dim = 0)
        joint_action = torch.cat((actions_p0.view(2*config.training.population_size,1),actions_p1.view(2*config.training.population_size,1)),1)
        joint_action = joint_action.type(torch.int8)
        next_obs, reward, done, info = envs.step(joint_action.cpu().numpy())

        obs_agent0.append(next_obs_p0)
        obs_agent1.append(next_obs_p1)
        

        next_obs_p0 = torch.Tensor(next_obs[:,0]).to(trainer.device)
        next_obs_p1 = torch.Tensor(next_obs[:,1]).to(trainer.device)
        next_done = torch.Tensor(done).to(trainer.device)

        cum_rew += reward

        if isinstance(trainer, HiPTTrainer) or isinstance(trainer, DIAYNTrainer) or isinstance(trainer, RIAYNTrainer):
            curr_p_len -= 1
    
    envs.close()
    state_embeddings_tensor = torch.stack(state_embeddings)  
    torch.save(state_embeddings_tensor, "state_embeddings.pt")
    all_z_step_tensor = torch.stack(all_z_step)
    torch.save(all_z_step_tensor, "all_z_step.pt")
    all_ent_step_tensor = torch.stack(entropy_z)
    torch.save(all_ent_step_tensor, "all_ent_step.pt")



    torch.save(torch.stack(obs_agent0) , "all_obs_agent0")
    torch.save(torch.stack(obs_agent0) , "all_obs_agent1")


    # all_preds_step_tensor = torch.stack(all_preds_step)
    # torch.save(all_preds_step_tensor, "all_disc_preds_step.pt")

    # print("Saved to", os.path.abspath("state_embeddings.pt"))

    # save_frames_as_gif(ep_visualizations[0], vis_path, f"/{config.layout.name}_blue_{config.model.name}_green_spagent{0}.gif")
    # print(f"Current z : {all_z_step_tensor[:,0]}")
    # print(f"Preds z : {all_preds_step_tensor[:,0]}")
     

    total_rew = 0
    total_rew_blue = 0
    total_rew_green = 0
    count_B = 0
    count_G = 0
    for ind, reward in enumerate(cum_rew):
        total_rew += reward
        if ind < config.training.population_size:
            print(f"Blue : {config.model.name} Agent, Green : SP Agent {ind}, Total reward: {reward}")
            total_rew_blue += reward
            count_B += 1
            # if config.evaluation.visualize_ep:
            #     save_frames_as_gif(ep_visualizations[ind], vis_path, f"/{config.layout.name}_blue_{config.model.name}_green_spagent{ind}.gif")   
        else:
            print(f"Blue : SP Agent {ind%config.training.population_size}, Green : {config.model.name} Agent, Total reward: {reward}")
            total_rew_green += reward
            count_G += 1
            # if config.evaluation.visualize_ep:
            #     save_frames_as_gif(ep_visualizations[ind], vis_path, f"/{config.layout.name}_blue_spagent{ind%config.training.population_size}_green_{config.model.name}.gif")
    
    # print(f"Total mean reward: {total_rew / len(cum_rew)} || Total reward as Blue agent: {total_rew_blue / count_B} || Total reward as Green agent: {total_rew_green / count_G}")