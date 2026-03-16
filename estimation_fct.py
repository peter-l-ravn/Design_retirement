import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
 

def prepare_data(par, year="all"):
    # Load data

    if year == "all":
        means_data = pd.read_csv("Data ny def/mean_matrix.csv")
        covariance_matrix = pd.read_csv("Data ny def/variance_matrix.csv")

    elif year == 2019:
        means_data = pd.read_csv("Data 2019/mean_matrix.csv")
        covariance_matrix = pd.read_csv("Data 2019/variance_matrix.csv")

    else:
        assert False

    # Process means
    assets = np.array(means_data["formue_plsats_Mean"])
    savings = np.array(means_data["pension_u_skat_plsats_Mean"])
    hours = np.array(means_data["yearly_hours_Mean"]) / par.full_time_hours
    extensive = np.array(means_data["extensive_v2_Mean"])
    hours = hours[:45]
    extensive = extensive[:45]

    mean = np.concatenate([extensive, assets, savings, hours])

    # Drop invalid moments
    covariance_matrix = pd.read_csv("Data ny def/variance_matrix.csv")
    variance_diag = np.diag(covariance_matrix.iloc[:,2:])
    variance_diag = variance_diag[~np.isnan(variance_diag)] 

    # Construct diagonal weighting matrix: 1/variance
    safe_variances = np.where(variance_diag == 0, 1e-6, variance_diag)  # Avoid divide-by-zero
    safe_variances[-55:] = safe_variances[-55:] / (par.full_time_hours**2)

    # safe_variances[:30] = safe_variances[:30] / 2

    weights = np.diag(1.0 / safe_variances)

    return mean, weights, {
        'extensive': extensive, 'assets': assets, 'savings': savings, 'hours': hours
    }


def load_and_process_data(mean_file, var_file, par, variables):
    # Load data
    means_data = pd.read_csv(mean_file)
    covariance_matrix = pd.read_csv(var_file)

    # Extract and process means
    mean_vectors = []
    for var in variables:
        mean_col = f"{var}_Mean"
        vec = np.array(means_data[mean_col])
        
        if var == "yearly_hours":
            vec = vec / par.full_time_hours
            vec = vec[:40]
        elif var == "extensive_v2":
            vec = vec[:40]

        mean_vectors.append(vec)

    mean = np.concatenate(mean_vectors)

    # Build row/column masks
    row_mask = covariance_matrix["_NAME_"].str.startswith(tuple(variables))
    col_mask = [col for col in covariance_matrix.columns if col.startswith(tuple(variables))]

    # Adjust for scale if hours included
    if "yearly_hours" in variables:
        hours_row_mask = covariance_matrix["_NAME_"].str.startswith("yearly_hours")
        hours_col_mask = [col for col in covariance_matrix.columns if col.startswith("yearly_hours")]

        covariance_matrix.loc[hours_row_mask, hours_col_mask] /= par.full_time_hours**2

    # Subset covariance matrix
    subset_matrix = covariance_matrix.loc[row_mask, col_mask]

    # Extract diagonal
    variance_diag = np.diag(subset_matrix)
    variance_diag = variance_diag[~np.isnan(variance_diag)]

    return mean, variance_diag



def scale_params(theta, bounds):
    """
    Scale theta to [0,1] given the bounds.
    """
    scaled = []
    for val, (low, high) in zip(theta, bounds):
        scaled_val = (val - low) / (high - low)
        scaled.append(scaled_val)
    return np.array(scaled)

def unscale_params(scaled_theta, bounds):
    """
    Convert scaled_theta in [0,1] back to original bounds.
    """
    unscaled = []
    for val, (low, high) in zip(scaled_theta, bounds):
        unscaled_val = low + val * (high - low)
        unscaled.append(unscaled_val)
    return np.array(unscaled)

def moment_func(sim_data):
    # Compute age-averaged moments
    avg_a_by_age = np.mean(sim_data.a, axis=0)  # Length 70
    # avg_s_by_age = np.mean(sim_data.s, axis=0)[:55]  # Length 70
    avg_h_by_age = np.nan_to_num(np.nanmean(np.where(sim_data.ex == 1, sim_data.h, np.nan), axis=0)[:40], nan=0.0) # Length 40
    avg_ex_by_age = np.mean(sim_data.ex, axis=0)[:40]  # Length 40

    # Concatenate and return
    return np.concatenate((avg_ex_by_age, avg_a_by_age, avg_h_by_age))


def simulate_moments(theta, theta_names, model):
        
    # 1. Update model parameters
    for i, name in enumerate(theta_names):
        setattr(model.par, name, theta[i])
    
    efterloen_options = [1, 0]
    flexible_hours_options = ["NVFI", "Fixed"]

    sim_means = []

    for efterloen in efterloen_options:
        for flexible_hours in flexible_hours_options:

            model.par.efter = efterloen
            model.par.flexible_hours = flexible_hours

            model.solve()
            model.simulate()

            sim_mean = [
                np.nanmean(np.where(model.sim.ex == 1, model.sim.h, np.nan), axis=0)[:45],
                np.mean(model.sim.ex, axis=0)[:45],
                np.mean(model.sim.a, axis=0),
                np.clip(np.mean(model.sim.s, axis=0), 0, None)
            ]

            sim_means.append(sim_mean)
    
    # 3. Return the expanded vector of simulated moments
    return sim_means

def obj_func(scaled_theta, theta_names, mom_data, W, model, bounds, do_print=False):
    start_time = time.time()  # Start timing

    theta = unscale_params(scaled_theta, bounds)

    if do_print: 
        print_str = ''
        for i, name in enumerate(theta_names):
            print_str += f'{theta[i]:2.10f}, '
        print(print_str)


    sim_means = simulate_moments(theta, theta_names, model)

    def loss_fn(params):

        hours_mean, extensive_mean, liquid_mean, _ = calc_means(sim_means, params)

        mom_sim = np.concatenate([extensive_mean[:40], liquid_mean, hours_mean[:40]])

        return (mom_data - mom_sim).T @ W @ (mom_data - mom_sim)
    

    def calc_means(sim_means, params):
        efterloen_share = params[0]
        flexible_share = params[1]

        p11 = efterloen_share * flexible_share
        p10 = efterloen_share * (1 - flexible_share)
        p01 = (1 - efterloen_share) * flexible_share
        p00 = (1 - efterloen_share) * (1 - flexible_share)

        hours_mean              = p11*sim_means[0][0] + p10*sim_means[1][0] + p01*sim_means[2][0] + p00*sim_means[3][0]
        extensive_mean          = p11*sim_means[0][1] + p10*sim_means[1][1] + p01*sim_means[2][1] + p00*sim_means[3][1]
        liquid_mean             = p11*sim_means[0][2] + p10*sim_means[1][2] + p01*sim_means[2][2] + p00*sim_means[3][2]
        illiquid_mean           = p11*sim_means[0][3] + p10*sim_means[1][3] + p01*sim_means[2][3] + p00*sim_means[3][3]

        return hours_mean, extensive_mean, liquid_mean, illiquid_mean

    x0 = np.array([0.5, 0.5])
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    res = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds,
                options={'ftol':1e-12, 'gtol':1e-8, 'maxiter':200})


    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    
    if do_print: 
        print(f"Error = {res.fun:.5f}, x = {res.x}, Time = {elapsed_time:.1f} seconds")

    return res.fun

