from pymoo.indicators.hv import HV
import numpy as np
from scipy.stats import norm
from openbox.utils.multi_objective import NondominatedPartitioning
from itertools import product
from scipy.optimize import minimize, brute, differential_evolution




def safe_divide(x1, x2):
    '''
    Divide x1 / x2, return 0 where x2 == 0
    '''
    return np.divide(x1, x2, out=np.zeros(np.broadcast(x1, x2).shape), where=(x2 != 0))


def acq_optimization(acq, args, bounds, acq_optimizer, seed):
    if acq=='hvi':
            acq_func = hvi_aq
    elif acq=='ehvi':
        acq_func = ehvi_aq

    n_lambdas = len(bounds) 
    
    if n_lambdas > 3:
        Ns = 3
    else:
        Ns = 20

    if acq_optimizer == 'brute':
        #rrange = [slice(bl,bu,0.01) for (bl,bu) in bounds]
        lam_final = brute(acq_func, bounds, Ns=Ns, args=args)#, finish=None)
    elif acq_optimizer == 'opt': 
        result = brute(acq_func, bounds, args=args, Ns=Ns, full_output=1)
        lam_final = result[0]
        best_result = result[1]
        for m in range(10):
            if m == 0:
                x0 = lam_final
            else:    
                x0 = np.zeros(n_lambdas) 
                for i in range(n_lambdas):  
                    x0[i] = np.random.uniform(bounds[i][0], bounds[i][1])
            fun = acq_func       
            result = minimize(fun, x0, args=args, bounds=bounds, tol=None, method='L-BFGS-B', options={'disp': False, 'maxiter':1000})
            if result.fun < best_result:
                best_result = result.fun
                lam_final = result.x
    elif acq_optimizer == 'de':
        result = differential_evolution(
                    acq_func,
                    bounds= bounds,
                    args= args,
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
        lam_final = result.x 
        
    for i, lam in enumerate(lam_final):  
        if lam < bounds[i][0]:
            lam = bounds[i][0]
        if lam > bounds[i][1]:
            lam = bounds[i][1]
        lam_final[i] = lam    
            
    return lam_final          


def hvi_aq(x, ref_point, curr_pfront, sur):
    x = np.round(x,2)
    f_new, _ = sur(x)
    ind = HV(ref_point=ref_point)                   
    curr_hv = ind(curr_pfront)
    new_hv = ind(np.vstack([curr_pfront, f_new]))
    #print(curr_hv-new_hv)
    return curr_hv-new_hv


def ehvi_aq(x, ref_point, curr_pfront, sur):
    acquisition_function = EHVI(ref_point)
    num_objectives = ref_point.shape[0]
    partitioning = NondominatedPartitioning(num_objectives, curr_pfront)
    cell_bounds = partitioning.get_hypercell_bounds(ref_point=ref_point)
    acquisition_function.update(cell_lower_bounds=cell_bounds[0],
                                cell_upper_bounds=cell_bounds[1])
    f_new, sig_new = sur(x) 

    acq_vals =  acquisition_function(f_new, sig_new)

    return -acq_vals                         


class EHVI:
    r"""Analytical Expected Hypervolume Improvement supporting m>=2 outcomes.

    This assumes minimization.

    Code is adapted from botorch. See [Daulton2020qehvi]_ for details.
    """

    def __init__(
        self,
        ref_point,
        **kwargs
    ):
        """Constructor

        Parameters
        ----------
        model: A fitted model.
        ref_point: A list with `m` elements representing the reference point (in the
            outcome space) w.r.t. to which compute the hypervolume. This is a
            reference point for the objective values (i.e. after applying
            `objective` to the samples).
        """
        #super().__init__(model=model, **kwargs)
        self.long_name = 'Expected Hypervolume Improvement'
        ref_point = np.asarray(ref_point)
        self.ref_point = ref_point
        self._cross_product_indices = np.array(
            list(product(*[[0, 1] for _ in range(ref_point.shape[0])]))
        )

    def update(self, **kwargs):
        """Update the acquisition functions values.

        This method will be called if the surrogate is updated. E.g.
        entropy search uses it to update its approximation of P(x=x_min),
        EI uses it to update the current optimizer.

        The default implementation takes all keyword arguments and sets the
        respective attributes for the acquisition function object.

        Parameters
        ----------
        kwargs
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])    

    def psi(self, lower: np.ndarray, upper: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        r"""Compute Psi function for minimization.

        For each cell i and outcome k:

            Psi(lower_{i,k}, upper_{i,k}, mu_k, sigma_k) = (
            sigma_k * PDF((upper_{i,k} - mu_k) / sigma_k) + (
            mu_k - lower_{i,k}
            ) * (1-CDF(upper_{i,k} - mu_k) / sigma_k)

        See Equation 19 in [Yang2019]_ for more details.

        Args:
            lower: A `num_cells x m`-dim array of lower cell bounds
            upper: A `num_cells x m`-dim array of upper cell bounds
            mu: A `batch_shape x 1 x m`-dim array of means
            sigma: A `batch_shape x 1 x m`-dim array of standard deviations (clamped).

        Returns:
            A `batch_shape x num_cells x m`-dim array of values.
        """
        u = (upper - mu) / sigma
        return sigma * norm.pdf(u) + (mu - lower) * (1 - norm.cdf(u))

    def nu(self, lower: np.ndarray, upper: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        r"""Compute Nu function for minimization.

        For each cell i and outcome k:

            nu(lower_{i,k}, upper_{i,k}, mu_k, sigma_k) = (
            upper_{i,k} - lower_{i,k}
            ) * (1-CDF((upper_{i,k} - mu_k) / sigma_k))

        See Equation 25 in [Yang2019]_ for more details.

        Args:
            lower: A `num_cells x m`-dim array of lower cell bounds
            upper: A `num_cells x m`-dim array of upper cell bounds
            mu: A `batch_shape x 1 x m`-dim array of means
            sigma: A `batch_shape x 1 x m`-dim array of standard deviations (clamped).

        Returns:
            A `batch_shape x num_cells x m`-dim array of values.
        """
        return (upper - lower) * (1 - norm.cdf((upper - mu) / sigma))

    def __call__(self, mu, sigma):
        #num_objectives = len(self.model)
        # mu = np.zeros((X.shape[0], 1, num_objectives))
        # sigma = np.zeros((X.shape[0], 1, num_objectives))
        # for i in range(num_objectives):
        #     mean, variance = self.model[i].predict_marginalized_over_instances(X)
        #     sigma[:, :, i] = np.sqrt(variance)
        #     mu[:, :, i] = -mean

        mu = -np.expand_dims(mu, axis=1)   
        sigma = np.expand_dims(sigma, axis=1)     
  

        cell_upper_bounds = np.clip(-self.cell_lower_bounds, -1e8, 1e8)

        psi_lu = self.psi(
            lower=-self.cell_upper_bounds,
            upper=cell_upper_bounds,
            mu=mu,
            sigma=sigma
        )
        psi_ll = self.psi(
            lower=-self.cell_upper_bounds,
            upper=-self.cell_upper_bounds,
            mu=mu,
            sigma=sigma
        )
        nu = self.nu(
            lower=-self.cell_upper_bounds,
            upper=cell_upper_bounds,
            mu=mu,
            sigma=sigma
        )
        psi_diff = psi_ll - psi_lu

        # This is batch_shape x num_cells x 2 x m
        stacked_factors = np.stack([psi_diff, nu], axis=-2)

        def gather(arr, index, axis):
            data_swaped = np.swapaxes(arr, 0, axis)
            index_swaped = np.swapaxes(index, 0, axis)
            gathered = np.choose(index_swaped, data_swaped)
            return np.swapaxes(gathered, 0, axis)

        # Take the cross product of psi_diff and nu across all outcomes
        # e.g. for m = 2
        # for each batch and cell, compute
        # [psi_diff_0, psi_diff_1]
        # [nu_0, psi_diff_1]
        # [psi_diff_0, nu_1]
        # [nu_0, nu_1]
        # This array has shape: `batch_shape x num_cells x 2^m x m`
        indexer = np.broadcast_to(self._cross_product_indices, stacked_factors.shape[:-2] + self._cross_product_indices.shape)
        all_factors_up_to_last = gather(stacked_factors, indexer, axis=-2)

        # Compute product for all 2^m terms, and sum across all terms and hypercells
        return all_factors_up_to_last.prod(axis=-1).sum(axis=-1).sum(axis=-1).reshape(-1, 1)

if __name__ == '__main__':
    ref_point = [0.0,0.0]
    acquisition_function = EHVI(ref_point)
    num_objectives = 2
    Y = np.random.randn(10,num_objectives)
    partitioning = NondominatedPartitioning(num_objectives, Y)
    cell_bounds = partitioning.get_hypercell_bounds(ref_point=ref_point)
    acquisition_function.update(cell_lower_bounds=cell_bounds[0],
                                     cell_upper_bounds=cell_bounds[1])    
