from scipy.optimize import root, brute, differential_evolution, minimize
import numpy as np


def compute_ref_point(lower_risk_n, upper_risk_n, bounds, sur, ref_optimizer):
    n_lambdas = len(bounds) 
    if n_lambdas> 3:
        Ns = 3
    else:
        Ns = 20
    def cross(x):
        f, _  = sur(x)
        return (f[0][:-1] - lower_risk_n).squeeze()

    if ref_optimizer == 'brute':
        lam = brute(lambda x: np.linalg.norm(cross(x)), bounds, Ns=Ns) 

    elif ref_optimizer == 'opt':   
        fun = lambda x: np.linalg.norm(cross(x))      
        result = brute(fun, bounds, Ns=Ns, full_output=1)
        lam = result[0]
        best_result = result[1]
        for m in range(10):
            if m == 0:
                x0 = lam
            else:    
                x0 = np.zeros(n_lambdas) 
                for i in range(n_lambdas):  
                    x0[i] = np.random.uniform(bounds[i][0], bounds[i][1])
              
            result = minimize(fun, x0, bounds=bounds, tol=None, method='L-BFGS-B', options={'disp': False, 'maxiter':1000})
            if result.fun < best_result:
                best_result = result.fun
                lam = result.x 

    elif ref_optimizer == 'const':   
        constraints = [{'type': 'ineq', 'fun': lambda x: sur(x)[0][0][i]-lower_risk_n[i]} for i in range(lower_risk_n.shape[0])]  # Example constraint: x + y >= 1

        fun = lambda x: sur(x)[0][0][-1]      
        result = brute(lambda x: np.linalg.norm(cross(x)), bounds, Ns=Ns, full_output=1)
        lam = result[0]
        best_result = result[1]
        for m in range(10):
            if m == 0:
                x0 = lam
            else:    
                x0 = np.zeros(n_lambdas) 
                for i in range(n_lambdas):  
                    x0[i] = np.random.uniform(bounds[i][0], bounds[i][1])
              
            result = minimize(fun, x0, bounds=bounds, constraints=constraints)
            if result.fun < best_result:
                best_result = result.fun
                lam = result.x               

    elif ref_optimizer == 'de':    

        result = differential_evolution(
                lambda x: np.linalg.norm(cross(x)),
                bounds= bounds,
                strategy="best1bin",
                maxiter=1000,
                popsize=50,
                tol=0.01,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=500,
                polish=True,
                callback=None,
                disp=False,
                init="latinhypercube",
                atol=0,
                ) 
        lam = result.x

    # #res = brute(lambda x: np.linalg.norm(cross(x)), bounds, Ns=5)
    # #res = root(cross, x0=np.random.uniform(size=(10)))


    ref_last, _ = sur(lam)
    ref_last = ref_last.squeeze()[-1]  
    ref_point = np.hstack((upper_risk_n, ref_last))

    return ref_point

    
