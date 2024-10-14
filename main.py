import torch
import random
import numpy as np
import os
import json

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import qmc


from task import from_task
from testing import FST, ideal_interval
from moo import is_pareto, filter_pareto_and_order, acq_optimization, run_parego, compute_ref_point
from utils import plot_init_configs, plot_BO, plot_res, res_to_df, factor_int, set_seed

front_types = {'Proposed': 'roi', 'HVI': 'full', 'EHVI': 'full', 'Upper Limit': 'upper'} 
acqs = {'Proposed': 'hvi', 'HVI': 'hvi', 'EHVI': 'ehvi', 'Upper Limit': 'hvi'}


def initialize(init_configs, start_vals, stop_vals, N, config0, init_type = 'Uniform', config=None):
    n_lambdas = len(start_vals)

    lambdas0 = np.array(config0).reshape(1,-1)
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])



    if init_type == 'LHS':
        sampler = qmc.LatinHypercube(d=n_lambdas, seed=config['seed'])
        lambdas = sampler.random(n=N-1)
        for i in range(n_lambdas):  
            lambdas[:,i] = start_vals[i] + lambdas[:,i]*(stop_vals[i]-start_vals[i])
        lambdas = np.concatenate((lambdas0, lambdas), axis=0)
    
    elif init_type == 'Uniform':
        Ns = factor_int(N,len(start_vals))
        Nrand = N-np.prod(Ns)
        lambads_random = np.zeros((Nrand, n_lambdas))
        lambdas_list = []
        for i, (start, stop, Ni) in enumerate(zip(start_vals, stop_vals, Ns)):
            lambdas_list.append(np.linspace(start, stop, Ni))
            lambads_random[:,i] = np.random.uniform(start_vals[i], stop_vals[i], Nrand)

        out = np.meshgrid(*lambdas_list)
        lambdas = np.concatenate(tuple([o.reshape(-1,1) for o in out]), axis=1)
        if np.any(np.all(lambdas0 == lambdas, axis=1)):
            lambdas = np.concatenate((lambdas, lambads_random), axis=0)
        else:
            lambdas = np.concatenate((lambdas0, lambdas, lambads_random[:-1,:]), axis=0)    


    elif init_type == 'Random':
        lambdas = np.zeros((N-1,n_lambdas))
        for i in range(n_lambdas):  
                lambdas[:,i] = np.random.uniform(start_vals[i], stop_vals[i], N-1)
        lambdas = np.concatenate((lambdas0, lambdas), axis=0)

    lambdas = np.round(lambdas,config["n_round"]) 

    objs = init_configs(lambdas, config)
    return lambdas, objs     



##########################
# ##  BBO
# #########################     


def Bayes_opt(train, lambdas_init, objs_init, N, front_type, acq, fixed_ref_point, config, lower_risks=None, upper_risks=None):
    n_tasks = objs_init.shape[1]

    length_scale = config["length_scale"]
    nu = config["nu"]
    alpha = config["alpha_gp"] 
    n_restarts_optimizer = config["n_restarts_optimizer"]

    m52 = ConstantKernel(1.0) * Matern(length_scale=length_scale, length_scale_bounds=(1e-10, 1e+5), nu=nu)

    model_list = [GaussianProcessRegressor(
                                        kernel=m52, 
                                        alpha=alpha, 
                                        n_restarts_optimizer=n_restarts_optimizer
                                    )
                    for _ in range(n_tasks)] 
    
    n_tasks = objs_init.shape[1]
    lambdas = lambdas_init
    g = objs_init 
    N0 = objs_init.shape[0]

    for n in range(N):
        print(f'Bayes opt - iteration {n+1}, out of {N}')

        ## Normalize objs
        transformation = StandardScaler()
        transformation.fit(g)
        g_norm = transformation.transform(g) 
        
        ## Fit ##
        for i in range(n_tasks):         
            model_list[i].fit(lambdas, g_norm[:,i])    

        n_lambdas = lambdas.shape[1]     

        def sur(x):
            x = x.reshape(-1,n_lambdas)
            mu, sigma = np.zeros((x.shape[0],n_tasks)), np.zeros((x.shape[0],n_tasks))
            for i in range(n_tasks):
                mu[:,i], sigma[:,i] = model_list[i].predict(x, return_std=True) 

            mu_unorm = transformation.inverse_transform(mu)
            sigma_unorm = transformation.inverse_transform(sigma)
            return mu_unorm, sigma_unorm 

        bounds = [(config["start_vals"][i], config["stop_vals"][i]) for i in range(n_lambdas)]

        if front_type == 'full':
            ref_point = fixed_ref_point
        elif front_type == 'roi': 
            ref_point = compute_ref_point(lower_risks[n,:], upper_risks[n,:], bounds, sur, config['ref_optimizer'])
            print(f'ref_point={ref_point}')          
        elif front_type == 'upper':
            ref_point = np.hstack((np.array(upper_risks), fixed_ref_point[-1]))
           
        curr_points, _ = sur(lambdas)
        curr_pfront = curr_points[is_pareto(curr_points),:]

        
        args = (ref_point, curr_pfront, sur)
             
        lam = acq_optimization(acq, args, bounds, config['acq_optimizer'], config['seed']+n)
        lambdas_next = np.round(lam, config["n_round"])
              
        f_select, _ = sur(lambdas_next)
        print(f'lambda_next = {lambdas_next}')
        print(f'f_select = {f_select}')

        ## Evaluate selected ##
        val_objs = train(lambdas_next, config)
        g_next = val_objs
        print(f'objs_next = {g_next}')
        
        ## Update pull ##
        lambdas = np.vstack((lambdas, lambdas_next.reshape(1,-1)))
        g = np.vstack((g, g_next.reshape(1,-1)))
        
    return lambdas, g, ref_point      


def test_path(evaluate_path, lambdas_path, objs_path, alphas, delta, binary_objs, random_states, config):

    n_objs = len(alphas)+1
    n_trials = random_states.shape[0]
    test_objs = np.zeros((n_trials, n_objs))            
    
    for t in range(n_trials):
        e_cal, e_test, sizes = evaluate_path(random_states[t], lambdas_path, config)
        
        print(e_cal)
        valid_lambdas = FST(e_cal, alphas, delta, sizes, binary_objs = binary_objs) 

        if len(valid_lambdas)>0:
            score = [e_cal[r,-1] + objs_path[r,-1] for r in valid_lambdas]
            id_min = score.index(min(score))
            print(len(valid_lambdas), id_min)

            id_select = valid_lambdas[id_min]
            test_objs[t,:] = e_test[id_select,:]
            print(objs_path[id_select,-1], e_cal[id_select,-1], e_test[id_select,-1])
        else:
            print('Failed')
            test_objs[t,:] = None

    return test_objs    



def main(config, methods, seed):
    
    config.update({'seed' : seed})
    config.update({'methods': methods})
     
    init_name = config["init_name_res_dir"]
    task_name = config["task_name"]
    delta_prime_start_left = config["delta_prime_start_left"]
    delta_prime_start_right = config["delta_prime_start_right"]
    delta_prime_end_left = config["delta_prime_end_left"]
    delta_prime_end_right = config["delta_prime_end_right"]
    n_trials = config["n_trials"]
    N0 = config["N0"]
    N_total = config["N_total"] 
    start_vals = config["start_vals"]
    stop_vals = config["stop_vals"]
    delta = config['delta']
    alphas_list = config['alphas_list']
    dataset = config["dataset"]
    init_type = config["init_type"]
    standard_ref_point = np.array(config["standard_ref_point"])
    objs_names = config["objs_names"]
    binary_objs = config["binary_objs"]
    config0 = config["config0"]
    n_val = config["n_val"]
    
    set_seed(seed)
    

    
    folder = f'results_{task_name}/seed_{seed}/{init_name}_dprime_s_{delta_prime_start_left}_{delta_prime_start_right}_e_{delta_prime_end_left}_{delta_prime_end_right}_trials_{n_trials}'

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    train, evaluate_path, init_configs, get_eff_sizes = from_task(task_name) 
    
    
    

    n_alphas = len(alphas_list)    
    n_methods = len(methods)

    results = {method: np.zeros((len(alphas_list), n_trials, len(objs_names))) for method in methods}

    np.random.seed(seed)
    random_states = np.random.permutation(n_trials)

    lambdas_selected = {}
    objs_selected = {}
    
    for method in (set(methods)& set(["Uniform", "Random"])):
        lambdas_selected[method], objs_selected[method] = initialize(init_configs, start_vals, stop_vals, N_total, config0, init_type = method, config=config)
        plot_init_configs(folder, method, objs_names, lambdas_selected[method], objs_selected[method])

    if 'Proposed' in methods or 'Upper Limit' in methods or 'ParEGO' in methods or 'HVI' in methods or 'EHVI' in methods:
        lambdas_init, objs_init = initialize(init_configs, start_vals, stop_vals, N0, config0, init_type = init_type, config=config)
        plot_init_configs(folder, 'BO', objs_names, lambdas_init, objs_init)
        print(lambdas_init)
        print(objs_init)
    
   
    if 'ParEGO' in methods:
        lambdas_selected['ParEGO'], objs_selected['ParEGO'] = run_parego(task_name, start_vals, stop_vals, len(objs_names), N_total, N0, config0, config)
        plot_init_configs(folder, 'ParEGO', objs_names, lambdas_selected['ParEGO'], objs_selected['ParEGO'])
    
   
    for method in (set(methods)& set(['HVI', 'EHVI'])):
        front_type = front_types[method]
        acq = acqs[method]

        print('-----------------------------------')
        print(f'Perfomring BO for {method}')
        print('-----------------------------------')
        
        lambdas_selected[method], objs_selected[method], ref_point = Bayes_opt(train,
                                                                lambdas_init, 
                                                                objs_init, 
                                                                N_total-N0, 
                                                                front_type, 
                                                                acq,
                                                                standard_ref_point,
                                                                config)
        
        plot_BO(folder, method, objs_names, objs_selected[method], N0, front_type, acq)

   
    for a, alphas in enumerate(alphas_list):
        print('-----------------------------------')
        print('-----------------------------------')
        print('i, alpha = ' ,a, alphas)
        print('-----------------------------------')
        print('-----------------------------------')
        for method in methods:

            if method in ["Proposed", "Upper Limit"]:
                front_type = front_types[method]
                acq = acqs[method]
                if method == "Proposed":
                    lower_risks, upper_risks = np.zeros((N_total-N0,len(alphas))), np.zeros((N_total-N0,len(alphas)))
                    factor_left = (delta_prime_end_left[0]/delta_prime_start_left[0])**(1/(N_total-N0-1)) 
                    factor_right = (delta_prime_end_right[0]/delta_prime_start_right[0])**(1/(N_total-N0-1)) 
                    for aa, alpha in enumerate(alphas):
                        for n in range(N_total-N0):
                            delta_prime_n = [delta_prime_start_left[0]*(factor_left**n),delta_prime_start_right[0]*(factor_right**n)] 
                            limit, lower_risks[n,aa], upper_risks[n,aa] = ideal_interval(alpha, delta, 
                                                                            n_cal=config['n_cal'],
                                                                            n_val=config['n_val'], 
                                                                            testing_method='FST',
                                                                            delta_prime=delta_prime_n,
                                                                            binary_obj=binary_objs[aa])
                            lower_risks[n,aa] = max(np.min(objs_init[:,aa]),lower_risks[n,aa])
                            print(n, alpha, delta_prime_n, limit, lower_risks[n,aa], upper_risks[n,aa])
                else:
                    upper_risks = alphas
                    lower_risks = None

                print('-----------------------------------')
                print(f'Perfomring BO for {method}')
                print('-----------------------------------')               
                
                lambdas_selected[method], objs_selected[method], ref_point = Bayes_opt(train,
                                                                                    lambdas_init, 
                                                                                    objs_init, 
                                                                                    N_total-N0, 
                                                                                    front_type, 
                                                                                    acq,
                                                                                    standard_ref_point,
                                                                                    config,
                                                                                    lower_risks,
                                                                                    upper_risks)

                plot_BO(folder, method, objs_names, objs_selected[method], N0, alphas, ref_point, lower_risks, upper_risks)
                

            
            print('-----------------------------------')
            print(f'Testing {method}')
            print('-----------------------------------')
            
            sizes = get_eff_sizes(objs_selected[method], n_val)
            lambdas_to_test, objs_to_test = filter_pareto_and_order(lambdas_selected[method], objs_selected[method], alphas, sizes, binary_objs)
            results[method][a,:,:] = test_path(evaluate_path, lambdas_to_test, objs_to_test, alphas, delta, binary_objs, random_states, config) 
                     

    res_df = res_to_df(folder, task_name, objs_names, results, alphas_list, methods, n_trials)
    res_df = plot_res(folder, task_name, objs_names, res_df)
    with open(os.path.join(folder,'config.json'), 'w') as f:
        json.dump(config, f)
                
        

     
