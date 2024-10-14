import numpy as np
from testing import combined_p_val

def is_pareto(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any((costs[:i])>=c, axis=1)) and np.all(np.any((costs[i+1:])>=c, axis=1))
    return is_efficient


def filter_pareto_and_order(rays_path, g_path, alphas, sizes, binary_objs):
    is_efficient = is_pareto(g_path) 
    print('Number of efficient' ,(is_efficient).astype(int).sum())
    all_ids = np.arange(g_path.shape[0])
    efficient_ids = all_ids[is_efficient]
    if g_path.shape[1] == 2:
        efficent_sorted = efficient_ids[np.argsort(g_path[efficient_ids,0])] 
    else:
        p_vals = combined_p_val(g_path[efficient_ids,:], alphas, sizes[efficient_ids,:], binary_objs)
        efficent_sorted = efficient_ids[np.argsort(p_vals)] 
    
    rays_to_test =  rays_path[efficent_sorted]  
    g_to_test = g_path[efficent_sorted,:]
    
    return rays_to_test, g_to_test
