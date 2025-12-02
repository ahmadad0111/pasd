# Adaptive Human–AI Coordination via Hierarchical Action Disentanglement


## Results

Below are visualizations showing **partner-skill evolution** during evaluation in two layouts: **Cramped Room** and **Forced Coordination**.

- **Cramped Room Evaluation**  
  ![Cramped Room](plots/cramped_room_skill_evolution.png)

  - **Gameplay GIF**  
  ![Cramped Room Gameplay](Results/10M/IAD/cramped_room/visualizations/cramped_room_blue_riayn_green_spagent0.gif)


- **Forced Coordination Evaluation**  
  ![Forced Coordination](plots/forced_coordination_skill_evolution.png)

  ![Forced Coordination Gameplay](Results/10M/IAD/forced_coordination/visualizations/forced_coordination_blue_riayn_green_spagent0.gif)

These visualizations illustrate how different partners evolve **distinct skill patterns** over time while maintaining dominant skills, highlighting **human-AI coordination dynamics**.

  


### Installation
* Create a new environment.
```
conda create -n hipt python=3.10
conda activate hipt
```
* Setup Overcooked AI environement. This repo uses an older commit of the [Overcooked AI](https://github.com/HumanCompatibleAI/overcooked_ai) repo. Make sure you are pulling from the correct version. 

```
git clone https://github.com/HumanCompatibleAI/overcooked_ai.git
cd overcooked_ai
git checkout 16f9428d99d9002be6611f3bab48f1bfe5c74c32
```

* Install dependencies.
```
./install.sh
```

### Training

* Note: This project uses Weights and Biases for logging. To enable logging, create a free account at [Weights and Biases](https://wandb.ai/). Then run ```wandb login``` and enter your API key. If you would like to disable tracking, set ```track=False``` .

To train a population of Self-Play Agents run the following command:
```
python main.py model=population layout={layout_name}
```

To train a IAD/HiPT/FCP agent with an existing SP Population run the following command:
```
python main.py model={hipt/fcp} layout={layout_name} layout.partner_pop_path={path_to_sp_population}
```
where ```layout_name``` is the name of the layout to train on and ```path_to_sp_population``` is the path to the SP population to use.

### Evaluation

To enable evaluation after training an agent/agent population simply set ```eval=True``` at the end of the training command. For example:
```
python main.py model=population eval=True #Plots the crossplay heatmap matrix

python main.py model={hipt/fcp} layout={layout_name} layout.partner_pop_path={path_to_sp_population} eval=True #Evaulates the HiPT/FCP agent on a SP Population and generates gifs of the gameplay.
```
