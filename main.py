import os
import torch
import random
import numpy as np
import wandb
import hydra
from hydra.utils import instantiate
import omegaconf
from omegaconf import open_dict
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from trainers.hipt_trainer import HiPTTrainer
from trainers.hipt_dynamic_trainer import HiPTTrainer_dyn
from trainers.diayn_trainer import DIAYNTrainer
from trainers.riayn_trainer import RIAYNTrainer
from trainers.fcp_trainer import FCPTrainer
from trainers.population_trainer import SPPopulationTrainer
from evaluation.evals import eval_rollout, plot_sp_heatmap, eval_rollout_dyn, eval_rollout_diayn, eval_rollout_diayn1


@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    #run_name = f"{config.gym_id}__{config.layout.name}__{config.model.name}__{config.seed}__{os.getcwd().split('/')[-2]}__{os.getcwd().split('/')[-1]}"
    run_name = f"{config.gym_id}__{config.layout.name}__{config.model.name}__{config.seed}__{os.path.basename(os.path.dirname(os.getcwd()))}__{os.path.basename(os.getcwd())}"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    if config.logging.track:        
        config_dict = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )
        wandb.init(
            project=config.logging.wandb_project_name,
            entity=config.logging.wandb_entity,
            sync_tensorboard=True,
            config = config_dict,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

        wandb.run.log_code(".")
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    torch.backends.cudnn.deterministic = config.torch_deterministic
    trainer = instantiate(config.model.trainer, config)
    

    if config.train:
        if isinstance(trainer,DIAYNTrainer):# or isinstance(trainer,RIAYNTrainer):
            trainer.train_s1()
        elif isinstance(trainer, RIAYNTrainer):
            print("Training RIAYN")
            trainer.train_combined()
        else:
            trainer.train()
    
    if isinstance(trainer,DIAYNTrainer):# or isinstance(trainer,RIAYNTrainer):
        if config.train_s2:
            if isinstance(trainer,DIAYNTrainer):
                #path = "C:/Users/adnan/Desktop/Overcooked_adnan/Results/diyan"
                path = os.path.join(os.path.dirname(base_dir),"hipt", "Results","10M", "diayn","s1", config.layout.name)
                print(f"Loading DIAYN model from path: {path}")
                #path = os.getcwd()

                trainer.load_model_state_diayn("best_agent",path)

            print("Training S2")
            trainer.train_s2()
            # trainer.train()

    print("Evaluting!")

    
    if config.eval:
        
        

        if isinstance(trainer, HiPTTrainer):
            # these two lie needs to be changed
            #path = "C:/Users/adnan/Desktop/Overcooked_adnan/Results/10M/hipt"
            path = os.path.join(os.path.dirname(base_dir),"hipt", "Results","10M", "hipt", config.layout.name)
            trainer.load_model_state("best_agent", path)

            eval_rollout(trainer, config)
        if isinstance(trainer, FCPTrainer):
            # these two lie needs to be changed
            #path = "C:/Users/adnan/Desktop/Overcooked_adnan/Results/10M/hipt"
            path = os.path.join(os.path.dirname(base_dir),"hipt", "Results","10M", "fcp", config.layout.name)
            trainer.load_model_state("best_agent", path)

            eval_rollout(trainer, config)

        if isinstance(trainer, HiPTTrainer_dyn):
            path = os.path.join(os.path.dirname(base_dir),"hipt", "Results","10M", "fcp", config.layout.name)
            trainer.load_model_state("best_agent", path)
            eval_rollout_dyn(trainer, config)
        if isinstance(trainer,DIAYNTrainer):
            path = os.path.join(os.path.dirname(base_dir),"hipt", "Results","10M", "diayn", config.layout.name)
            trainer.load_model_state_diayn("best_agent",path)
            eval_rollout_diayn(trainer, config)

        if isinstance(trainer,RIAYNTrainer): # PASD
            #path = "C:/Users/adnan/Desktop/Overcooked_adnan/Results/riyan"
            path = os.path.join(os.path.dirname(base_dir),"hipt", "Results","10M", "pasd", config.layout.name)
            trainer.load_model_state_diayn("best_agent",path)
            eval_rollout_diayn(trainer, config)
        elif isinstance(trainer, SPPopulationTrainer):
            path = os.path.join("sp_populoation", config.layout.name)
            plot_sp_heatmap(trainer, config, path)
            
if __name__ == "__main__":
    main()
