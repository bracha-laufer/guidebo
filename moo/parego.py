import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from matplotlib import pyplot as plt
from task import from_task

from ConfigSpace import ConfigurationSpace,Configuration
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.multi_objective.parego import ParEGO

def run_parego(task_name, start_vals, stop_vals, n_objs, N_total, N0, config0, alg_config):
    
    n_lambdas = len(start_vals)
    cs = ConfigurationSpace()
    for i, (start, stop) in enumerate(zip(start_vals, stop_vals)):
        cs.add_hyperparameter(UniformFloatHyperparameter(f"lam{i}", lower=start, upper=stop))

    train, _, _, _ = from_task(task_name) 

    def compute_config(config, seed=0):
       
        lambdas = np.round(np.array([config[f"lam{i}"] for i in range(n_lambdas)]),alg_config["n_round"])

        print(lambdas)
        val_scores = train(lambdas, alg_config)
        print(val_scores)

        return {f"metric{i}": score for i,score in enumerate(val_scores)}


    # Define our environment variables
    scenario = Scenario(
        cs,
        objectives=[f"metric{i}" for i in range(n_objs)],
    #    walltime_limit=30,  # After 30 seconds, we stop the hyperparameter optimization
        n_trials=N_total,  # Evaluate max 200 different trials
        n_workers=1,
    )

    #configs_init = [Configuration(cs, vector=np.array(lam)) for lam in lambdas_init]
    # if task_name != 'vae': # got stuck when running with configs_init so skip
    #     initial_design = HPOFacade.get_initial_design(scenario, n_configs=0, additional_configs=configs_init)
    # else:
    #print(config0)
    #print(Configuration(cs, values={f"lam{i}":config0[i] for i in range(n_lambdas)}))
    initial_design = None
    
    #configs_init = [Configuration(cs, values={f"lam{i}":lam[i] for i in range(n_lambdas)}) for lam in lambdas_init]
    configs_init = [Configuration(cs, values={f"lam{i}":config0[i] for i in range(n_lambdas)})]
    initial_design = HPOFacade.get_initial_design(scenario, n_configs=0, additional_configs=configs_init)
    print(initial_design)
        
    multi_objective_algorithm = ParEGO(scenario)
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)

    # Create our SMAC object and pass the scenario and the train method
    smac = HPOFacade(
        scenario,
        compute_config,
        initial_design=initial_design,
        multi_objective_algorithm=multi_objective_algorithm,
        intensifier=intensifier,
        overwrite=True,
    )

    # Let's optimize
    incumbents = smac.optimize()

    rh = smac.runhistory
    objs = np.vstack([o.cost for o in rh.values()])
    config_ids = [i.config_id for i in rh.keys()]
    configs = [rh.get_config(config_id) for config_id in config_ids]

    selected_configs = np.zeros((len(configs),n_lambdas))
    for i in range(n_lambdas):
       selected_configs[:,i]  = np.round(np.array([config[f"lam{i}"] for config in configs]),alg_config["n_round"])
    
    #selected_configs = np.concatenate((config0, selected_configs), axis=0)        
    print(selected_configs)
    return selected_configs, objs
