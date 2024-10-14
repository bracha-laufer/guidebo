
import numpy as np
from .bounds import hb_p_value_left_tail, binom_p_value_left_tail, hb_p_value_right_tail, binom_p_value_right_tail

def combined_p_val(Qs, alphas, sizes, binary_objs):
    Qs = np.maximum(Qs,0)
    c = len(alphas)
    if binary_objs[0]:
       p_vals = binom_p_value_left_tail(Qs[:,0], sizes[:,0], alphas[0])
    else:    
       p_vals= hb_p_value_left_tail(Qs[:,0], sizes[:,0], alphas[0])
    
    for i in range(1,c):
        if binary_objs[i]:
            p_vals_i = binom_p_value_left_tail(Qs[:,i], sizes[:,i], alphas[i])
        else:    
            p_vals_i = hb_p_value_left_tail(Qs[:,i], sizes[:,i], alphas[i])
        p_vals = np.maximum(p_vals, p_vals_i)

    return p_vals 

def FST(error_cal, alpha, delta, sizes, binary_objs): 
    p_vals = combined_p_val(error_cal, alpha, sizes, binary_objs)
    n_rays = error_cal.shape[0]
    
    valid_rays = []
    for r in range(n_rays):
        if p_vals[r] < delta:
            valid_rays.append(r)
        else: 
            break    

    return valid_rays   

def ideal_interval(alpha, delta, n_cal, n_val, testing_method, delta_prime=[0.01,0.01], binary_obj=False, T=10):
    if binary_obj:
       p_func = binom_p_value_left_tail
    else:    
       p_func = hb_p_value_left_tail
    
    R1 = np.linspace(0,alpha,10000)
    p_vals = p_func(R1, n_cal, alpha)
    if testing_method == 'Bonferroni':
        id_max = np.argmax(p_vals>=delta/T)
    elif testing_method == 'FST':
        id_max = np.argmax(p_vals>=delta)    

    p_vals = p_func(R1, n_val, R1[id_max-1])
    id_lower = np.argmax(p_vals>=delta_prime[0])

    if binary_obj:
       p_func = binom_p_value_right_tail
    else:    
       p_func = hb_p_value_right_tail

    R2 = np.linspace(R1[id_max-1],1.0,10000)
    p_vals = p_func(R2, n_val, R1[id_max-1])
    id_upper = np.argmin(p_vals>=delta_prime[1])   

    return R1[id_max-1], R1[id_lower-1], R2[id_upper]



