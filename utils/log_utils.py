import os
import numpy as np
import pandas as pd

def res_to_df(folder, task_name, scores_names, results, alphas_list, methods, n_trials):

    n_alphas = len(alphas_list)
    n_methods = len(methods)
    n_scores = len(scores_names)

    flat_methods = np.repeat(methods, n_trials*n_alphas)

    res_df = pd.DataFrame({
        'Method': flat_methods
    })
    
    for i in range(n_scores):
        flat_scores = np.concatenate([v[:,:,i].reshape(-1) for v in results.values()], axis=0)
        res_df[scores_names[i]] = flat_scores
    
    for i in range(n_scores-1):
        alphas_list_i = [a[i] for a in alphas_list]
        flat_alphas = np.repeat(alphas_list_i, n_trials).tolist()*n_methods
        res_df[f'$\\alpha_{i+1}$'] = flat_alphas

    res_df.to_csv(os.path.join(folder,f'results_{task_name}.csv'))

    return res_df

def res_to_df_per_budget(folder, task_name, scores_names, results, alphas, min_budget, max_budget, methods, n_trials):

    n_budgets = max_budget - min_budget + 1
    n_methods = len(methods)
    n_scores = len(scores_names)

    flat_methods = np.repeat(methods, n_trials*n_budgets)

    res_df = pd.DataFrame({
        'Method': flat_methods
    })
    
    for i in range(n_scores):
        flat_scores = np.concatenate([v[:,:,i].reshape(-1) for v in results.values()], axis=0)
        res_df[scores_names[i]] = flat_scores
    
    
    
    for i in range(n_scores-1):
        res_df[f'$\\alpha_{i+1}$'] = alphas[i]
        
    budgets = np.arange(min_budget, max_budget+1)
    flat_budgets = np.repeat(budgets, n_trials).tolist()*n_methods
    res_df['Budget'] = flat_budgets

    res_df.to_csv(os.path.join(folder,f'results_{task_name}_per_budget.csv'))

    return res_df

