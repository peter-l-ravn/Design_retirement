from numba import njit, prange
import numpy as np 

from consav.linear_interp import interp_1d, interp_2d, interp_3d, interp_4d
from optimizers import optimizer, optimize_outer, interp_3d_vec, optimizer_with_start
from jit_module import jit_if_enabled

import math

from consav_grids import nonlinspace

#######################################################################
# Structure 
#   1. Essentiel functions for such as utility, bequest, and wage
#   2. Value functions
#   3. Helper functions in solving and optimizing
#   4. Objective functions 
#   5. Solving the model
#######################################################################


# 1. Essentiel functions for such as utility, bequest, and wage 
@jit_if_enabled(fastmath=False)
def utility(par, c, h, k, t):
    work_dummy = 1 if (t>=par.retirement_age and h==0) else 0
    # return ((c+1)**(1-par.sigma))/(1-par.sigma) + par.dummy*work_dummy - (par.zeta/(1+k)) * (h**(1+par.gamma))/(1+par.gamma) - par.gamma_1*h*t**2
    return ((c+1)**(1-par.sigma))/(1-par.sigma) - (par.zeta/(1+k)) * (h**(1+par.gamma))/(1+par.gamma) - par.gamma_1*h*(np.exp((t - par.retirement_age)/par.gamma_2))/(1 + np.exp((t - par.retirement_age)/par.gamma_2))

@jit_if_enabled(fastmath=False)
def utility_c(par, c):
    return ((c+1)**(1-par.sigma))/(1-par.sigma)


@jit_if_enabled(fastmath=False)
def bequest(par, a):
    return par.mu*(a+par.a_bar)**(1-par.sigma) / (1-par.sigma)

@jit_if_enabled(fastmath=False)
def wage(par, k, t):
    '''Wage before taxes'''
    if t <= 30:
        return par.full_time_hours*np.exp(np.log(par.w_0) + par.beta_1*k + par.beta_2*t**2)
    else:
        return par.full_time_hours*np.exp(np.log(par.w_0) + par.beta_1*k + par.beta_2*30**2)

# 1.1 The four sources of income all before taxes and retirement contributions - and total income before taxes and retirement contributions:
# 1.1.1 Capital income
@jit_if_enabled(fastmath=False)
def capital_return_fct(par, a):
    '''Capital return is the same for all periods'''
    return (par.r_a/(1+par.r_a)) * a

# 1.1.2. Retirement payouts
@jit_if_enabled(fastmath=False)
def calculate_retirement_payouts(par, h, s, e, r, t):
    """Calculate retirement payouts: can be split into 3 periods: before retirement, during installment and annuity, and only annuity"""
    
    if (e == 2) or (t >= par.first_retirement and h == 0.0):
        if t >= r + par.m:
            s_retirement = s
            s_lr = s_retirement * par.share_lr * par.s_lr_deterministic[int(r)]
            return  s_lr, 0.0
        
        elif t >= r:
            s_retirement = s
            s_lr = s_retirement * par.share_lr * par.s_lr_deterministic[int(r)]
            s_rp = s_retirement * (1-par.share_lr) * ( par.r_s * (1+par.r_s)**par.m ) / ( (1+par.r_s)**par.m - 1 )
            return   s_lr, s_rp
        
        else:
            print(e, t, r)
            print("Error: Invalid retirement payout calculation")
    else:
        return 0.0, 0.0

# 1.1.3. Labor income
@jit_if_enabled(fastmath=False)
def labor_income_fct(par, k, h, r, t):
    '''Before and after retirement age'''
    return h*wage(par, k, t)

# 1.1.4. Public benefits
@jit_if_enabled(fastmath=False)
def public_benefit_fct(par, h, e, ef, income, t):
    """Before retirement: unemployment benefits (if working, then no benefits), after retirement: public pension"""
    # Before public retirement age
    if t < par.first_retirement + par.early_benefits_lag:
        if h > 0.0:
            return 0.0
        elif e == par.emp or e == par.unemp:
            # Unemployment benefits
            return max(par.unemployment_benefit[t,0] - income, 0)
        elif e == par.ret:
            # Retirement benefits
            return par.early_benefit[t]
        else:
            print("Error: Invalid employment status")
            return par.unemployment_benefit[t][0]
    
    elif t < par.retirement_age:
        if ef == 1: # overførsel for efterløn
            if h > 0.0:
                return 0.0
            elif e == par.emp or e == par.unemp:
                return max(par.efterloen - income*par.rho_ef, 0)
            elif e == par.ret:
                # Retirement benefits
                return par.early_benefit[t]
        else: # overførsel, hvis ikke ret til efterløn
            if h > 0.0:
                return 0.0
            elif e == par.emp or e == par.unemp:
                # Unemployment benefits
                return max(par.unemployment_benefit[t,0] - income, 0)
            elif e == par.ret:
                # Retirement benefits
                return par.early_benefit[t]
    # public retirement benefits
    else:
        # return max(par.chi_base, par.chi_total - income*par.rho)
        return max(par.chi_base, par.chi_total - max(0, income - par.fradrag ) * par.rho )
    

    
# 1.1.5 Total income before taxes and retirement contributions
@jit_if_enabled(fastmath=False)
def income_private_fct(par, a, s, k, h, e, r, ef, t):
    '''Private income is taxed and is used for 
    Income before taxes and contribution: include capital return, retirement payouts, and wages'''
    # Capital income
    a_return = 0.0 # capital_return_fct(par, a), skal vi have capital return med?

    # Retirement payouts
    if (e == 2) or (t >= par.first_retirement and h == 0.0):
        s_lr, s_rp = calculate_retirement_payouts(par, h, s, e, r, t)  # par, h, s, e, r, t
    else:
        s_lr, s_rp = 0.0, 0.0
    
    # labor income 
    labor_income = labor_income_fct(par, k, h, r, t)

    # Total income 
    total_income = labor_income + a_return + s_lr + s_rp

    # public benefits
    public_benefit = public_benefit_fct(par, h, e, ef, total_income, t)

    return total_income + public_benefit


@jit_if_enabled(fastmath=False)
def tax_rate_fct(par, a, s, k, h, e, r, ef, t):
    total_income = income_private_fct(par, a, s, k, h, e, r, ef, t)

    labor_market_contribution = total_income * par.labor_market_rate
    personal_income = total_income - labor_market_contribution
    taxable_base = max(personal_income - par.personal_allowance, 0)
    employment_deduction = min(par.employment_deduction_cap, total_income * par.employment_deduction_rate)
    taxable_income = taxable_base - employment_deduction
    bottom_tax = taxable_base * par.bottom_tax_rate
    top_tax = par.top_tax_rate * max(personal_income - par.top_tax_threshold, 0)
    municipal_tax = par.municipal_tax_rate * taxable_income

    # Note: Church tax is not included in 'income_tax'
    income_tax = labor_market_contribution + bottom_tax + top_tax + municipal_tax
    income_after_tax = total_income - income_tax

    # Effective tax rate
    effective_tax_rate = max(1 - (income_after_tax / total_income), 0)
    return effective_tax_rate
    # return par.upsilon


# 1.2.2 retirement contributions, only of labor income 
@jit_if_enabled(fastmath=False)
def retirement_contribution_fct(par, a, s, k, h, r, t):
    '''Retirement contributions'''
    return labor_income_fct(par, k, h, r, t)*(par.tau_private[t] + par.tau_employer[t])


# 1.3. calculate income after taxes and contributions
# 1.3.1 Income after taxes and contributions
@jit_if_enabled(fastmath=False)
def final_income_and_retirement_contri(par, a, s, k, h, e, r, ef, t):
    ''' The following is taxed: capital income, retirement payouts, labor income, and public benefits
    retirement contributions are only of labor income'''
    # Incomes before taxes and contributions
    a_return = 0.0 # capital_return_fct(par, a), skal vi have capital return med?
    if (e == 2) or (t >= par.first_retirement and h == 0.0):
        s_lr, s_rp = calculate_retirement_payouts(par, h, s, e, r, t)
    else:
        s_lr, s_rp = 0.0, 0.0
    
    labor_income = labor_income_fct(par, k, h, r, t)
    income_private = a_return + s_lr + s_rp + labor_income
    chi = public_benefit_fct(par, h, e, ef, income_private, t)

    # Tax rate and retirement contribution
    tax_rate = tax_rate_fct(par, a, s, k, h, e, r, ef, t)
    retirement_contribution = retirement_contribution_fct(par, a, s, k, h, r, t)

    if h > 0.0:
        return (1-tax_rate)*(income_private*(1-par.tau_private[t]) + chi), retirement_contribution
    else:
        return (1-tax_rate)*(income_private + chi), retirement_contribution


# 2. Helper functions in solving and optimizing
@jit_if_enabled(fastmath=False)
def budget_constraint(par, h, a, s, k, e, r, ef, income, t):
    return par.c_min, max(par.c_min*2, a + income)


# @jit_if_enabled(fastmath=False)
# def compute_transitions(par, sol_V, employed, retirement_idx, ex_next, t):
    
#     if t == par.last_retirement:
#         V_next_em       = sol_V[t+1, :, :, :, retirement_idx, par.ret]
#         V_next_un       = sol_V[t+1, :, :, :, retirement_idx, par.ret]
#         V_next_early    = sol_V[t+1, :, :, :, retirement_idx, par.ret]

#     elif t >= par.retirement_age - 1: # Use expected value one year before the current period
#         if int(ex_next) == par.unemp:
#             V_next_em       = sol_V[t+1, :, :, :, retirement_idx+1, par.ret]
#         else:
#             V_next_em       = sol_V[t+1, :, :, :, retirement_idx+1, par.emp]

#         V_next_un       = sol_V[t+1, :, :, :, retirement_idx+1, par.ret]
#         V_next_early    = sol_V[t+1, :, :, :, retirement_idx+1, par.ret]

#     else:
#         V_next_em       = sol_V[t+1, :, :, :, retirement_idx+1, int(ex_next)]
#         V_next_un       = sol_V[t+1, :, :, :, retirement_idx+1, par.unemp]
#         V_next_early    = sol_V[t+1, :, :, :, retirement_idx+1, par.ret]

#     V_next = par.p_e_0[t]*V_next_un + par.p_e_1[t]*V_next_em + par.p_e_2[t] * V_next_early

#     return V_next


@jit_if_enabled(parallel=True)
def precompute_EV_next(par, sol_V, retirement_idx, employed, t):

    a_grid = par.a_grid
    s_grid = par.s_grid
    k_grid_t = par.k_grid[t]
    k_grid_tp1 = par.k_grid[t+1]

    Na = len(a_grid)
    Ns = len(s_grid)
    Nk = len(k_grid_t)

    xi_v = par.xi_v
    xi_p = par.xi_p
    N_xi = par.N_xi

    p0 = par.p_e_0[t]
    p1 = par.p_e_1[t]
    p2 = par.p_e_2[t]

    # pre-sliced continuation values at retirement_idx+1
    if t != par.last_retirement:
        V_next_ret = sol_V[t+1, :, :, :, retirement_idx+1, par.ret]
        V_next_emp = sol_V[t+1, :, :, :, retirement_idx+1, par.emp]
        V_next_unemp = sol_V[t+1, :, :, :, retirement_idx+1, par.unemp]
    else:
        pass

    # only used in the last-retirement case
    V_next_ret_same = sol_V[t+1, :, :, :, retirement_idx, par.ret]

    EV = np.zeros((Na, Ns, Nk))
    N_total = Na * Ns * Nk

    for flat in prange(N_total):
        i_a = flat // (Ns * Nk)
        rem = flat - i_a * (Ns * Nk)
        i_s = rem // Nk
        i_k = rem - i_s * Nk

        a_next = a_grid[i_a]
        s_next = s_grid[i_s]
        k_next = k_grid_t[i_k]

        EV_val = 0.0

        for idx in range(N_xi):
            k_temp = k_next * xi_v[idx]

            # determine ex_next
            if t == par.last_retirement:
                ex_next = par.ret

            elif t >= par.retirement_age:
                if employed == par.emp:
                    sol_v_unemp = interp_3d(a_grid, s_grid, k_grid_tp1, V_next_ret, a_next, s_next, k_temp)
                    sol_v_emp = interp_3d(a_grid, s_grid, k_grid_tp1, V_next_emp, a_next, s_next, k_temp)
                    ex_next = 1 if sol_v_emp >= sol_v_unemp else 0
                else:
                    ex_next = 0

            elif t >= par.first_retirement:
                if employed == par.emp:
                    sol_v_unemp = interp_3d(a_grid, s_grid, k_grid_tp1, V_next_unemp, a_next, s_next, k_temp)
                    sol_v_emp = interp_3d(a_grid, s_grid, k_grid_tp1, V_next_emp, a_next, s_next, k_temp)
                    ex_next = 1 if sol_v_emp >= sol_v_unemp else 0
                else:
                    ex_next = 0

            else:
                if employed == par.emp or employed == par.unemp:
                    sol_v_unemp = interp_3d(a_grid, s_grid, k_grid_tp1, V_next_unemp, a_next, s_next, k_temp)
                    sol_v_emp = interp_3d(a_grid, s_grid, k_grid_tp1, V_next_emp, a_next, s_next, k_temp)
                    ex_next = 1 if sol_v_emp >= sol_v_unemp else 0
                else:
                    ex_next = 0

            # choose relevant continuation arrays
            if t == par.last_retirement:
                V_em = V_next_ret_same
                V_un = V_next_ret_same
                V_er = V_next_ret_same

            elif t >= par.retirement_age - 1:
                V_em = V_next_ret if int(ex_next) == par.unemp else V_next_emp
                V_un = V_next_ret
                V_er = V_next_ret

            else:
                V_em = V_next_unemp if int(ex_next) == par.unemp else V_next_emp
                V_un = V_next_unemp
                V_er = V_next_ret

            v_em = interp_3d(a_grid, s_grid, k_grid_tp1, V_em, a_next, s_next, k_temp)
            v_un = interp_3d(a_grid, s_grid, k_grid_tp1, V_un, a_next, s_next, k_temp)
            v_er = interp_3d(a_grid, s_grid, k_grid_tp1, V_er, a_next, s_next, k_temp)

            EV_val += (p0 * v_un + p1 * v_em + p2 * v_er) * xi_p[idx]

        EV[i_a, i_s, i_k] = EV_val

    return EV

@jit_if_enabled(fastmath=False)
def calculate_last_period_consumption(par, a, s, e, r, t):
    k, h, ef = 0.0, 0.0, 0.0
    income, _ = final_income_and_retirement_contri(par, a, s, k, h, e, r, ef, t)
 
    if par.mu != 0.0:
        # With bequest motive
        return max(((1/(1+(par.mu*(1+par.r_a))**(-1/par.sigma)*(1+par.r_a))) 
                    * (par.mu*(1+par.r_a))**(-1/par.sigma) 
                    * ((1+par.r_a)*(a+income)+par.a_bar)), 0)
    
    else: 
        # No bequest motive
        return (a + income)


# 3. Value functions
@jit_if_enabled(fastmath=False)
def value_last_period(par, c, a, s, e, r, t):
    # states and income 
    h, k, ef = 0.0,0.0, 0.0
    income, _ = final_income_and_retirement_contri(par, a, s, k, h, e, r, ef, t)
    a_next = (1+par.r_a)*(a + income - c)

    return utility(par, c, h, k, t) + bequest(par, a_next)


@jit_if_enabled(fastmath=False)
def value_function_after_retirement(par, sol_V, c, a, s, e, r, ef, income, t):
    # states and income 
    retirement_age_idx = r
    if t >= par.retirement_age - 1:
        e_idx_next = par.ret
    else:
        e_idx_next = par.unemp

    h, k  = 0.0, 0.0
    k_idx = 0

    # Next period states 
    a_next = (1+par.r_a)*(a + income - c)
    s_next = s
    V_next = sol_V[t+1, :, :, k_idx, retirement_age_idx, e_idx_next]
    EV_next = interp_2d(par.a_grid, par.s_grid, V_next, a_next, s_next)

    return utility(par, c, h, k, t) + par.pi[t]*par.beta*EV_next + (1-par.pi[t])*bequest(par, a_next)


@jit_if_enabled(fastmath=False)
def value_function(par, sol_V, sol_EV, c, h, a, s, k, e, r, ef, income, retirement_contribution, t):

    # Next period states
    a_next = (1+par.r_a)*(a + income - c)
    s_next = (1+par.r_s)*(s + retirement_contribution)
    k_next = ((1-par.delta)*k + h)
    EV_next = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_EV, a_next, s_next, k_next)

    return utility(par, c, h, k, t) + par.pi[t]*par.beta*EV_next + (1-par.pi[t])*bequest(par, a_next)

@jit_if_enabled(fastmath=False)
def value_function_given_m(c, par, sol_EV, a_m, s_m, k_m, t):
    # Next period states
    a_next = (1+par.r_a)*(a_m - c)
    s_next = (1+par.r_s)*(s_m)
    k_next = ((1-par.delta)*k_m)
    EV_next = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_EV, a_next, s_next, k_next)

    return utility_c(par, c) + par.pi[t]*par.beta*EV_next + (1-par.pi[t])*bequest(par, a_next)

@jit_if_enabled(fastmath=False)
def obj_consumption_given_m(c, par, sol_EV, a_m, s_m, k_m, t):
    return -value_function_given_m(c, par, sol_EV, a_m, s_m, k_m, t)


@jit_if_enabled(fastmath=False)
def value_function_NVFI(h, par, sol_c_given_m, sol_EV, a, s, k, e, r, ef, t):

    income, retirement_contribution = final_income_and_retirement_contri(par, a, s, k, h, e, r, ef, t)

    a_m = a + income
    s_m = s + retirement_contribution
    k_m = k + h / (1 - par.delta)

    a_grid_m = par.a_m_grid[t, :, r, e]
    s_grid_m = par.s_m_grid[t, :, r, e]
    k_grid_m = par.k_m_grid[t, :, r, e]

    a_m = min(max(a_m, a_grid_m[0]), a_grid_m[-1])
    s_m = min(max(s_m, s_grid_m[0]), s_grid_m[-1])
    k_m = min(max(k_m, k_grid_m[0]), k_grid_m[-1])

    c_star = interp_3d(
        a_grid_m, s_grid_m, k_grid_m,
        sol_c_given_m[t, :, :, :, r, e],
        a_m, s_m, k_m
    )

    a_next = (1+par.r_a)*(a + income - c_star)
    s_next = (1+par.r_s)*(s + retirement_contribution)
    k_next = (1-par.delta)*k + h
    EV_next = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_EV, a_next, s_next, k_next)

    return utility(par, c_star, h, k, t) + par.pi[t]*par.beta*EV_next + (1-par.pi[t])*bequest(par, a_next)

# @jit_if_enabled(fastmath=False)
# def value_function_NVFI(h, par, sol_c_given_m, sol_EV, a, s, k, e, r, ef, t):
#     # states and income 
#     income, retirement_contribution = final_income_and_retirement_contri(par, a, s, k, h, e, r, ef, t)

#     # Next period states
#     a_m = a + income
#     s_m = s + retirement_contribution
#     k_m = k + (h / (1-par.delta))

#     c_star = interp_3d(par.a_m_grid[t, :, r, e], par.s_m_grid[t, :, r, e], par.k_m_grid[t, :, r, e], sol_c_given_m[t, :, :, :, r, e], a_m, s_m, k_m)

#     a_next = (1+par.r_a)*(a + income - c_star)
#     s_next = (1+par.r_s)*(s + retirement_contribution)
#     k_next = ((1-par.delta)*k + h)
#     EV_next = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_EV, a_next, s_next, k_next)

#     return utility(par, c_star, h, k, t) + par.pi[t]*par.beta*EV_next + (1-par.pi[t])*bequest(par, a_next)

@jit_if_enabled(fastmath=False)
def obj_hours_NVFI(h, par, sol_c_given_m, sol_EV, a, s, k, e, r, ef, t):
    return -value_function_NVFI(h, par, sol_c_given_m, sol_EV, a, s, k, e, r, ef, t)

# 4. Objective functions 
@jit_if_enabled(fastmath=False)
def obj_consumption(c, par, sol_V, sol_EV, h, a, s, k, e, r, ef, income, retirement_contribution, t):
    return -value_function(par, sol_V, sol_EV, c, h, a, s, k, e, r, ef, income, retirement_contribution, t)


@jit_if_enabled()
def obj_consumption_after_retirement(c, par, sol_V, a, s, e, r, ef, income, t):
    return -value_function_after_retirement(par, sol_V, c, a, s, e, r, ef, income, t)


@jit_if_enabled(fastmath=False)
def obj_hours(h, par, sol_V, sol_EV, a, s, k, e, r, ef, t, dist):

    income, retirement_contribution = final_income_and_retirement_contri(par, a, s, k, h, e, r, ef, t)

    bc_min, bc_max = budget_constraint(par, h, a, s, k, e, r, ef, income, t)
    
    c_star = optimizer(
        obj_consumption,     
        bc_min, 
        bc_max,
        args=(par, sol_V, sol_EV, h, a, s, k, e, r, ef, income, retirement_contribution, t),
        tol=dist
    )
    
    val_at_c_star = -value_function(par, sol_V, sol_EV, c_star, h, a, s, k, e, r, ef, income, retirement_contribution, t)
    
    return val_at_c_star

@jit_if_enabled(fastmath=False)
def optimizer_guess(guess, min, max):
    if np.isnan(guess):
        return (min + max) / 2
    else:
        return guess

# 5. Solving the model
@jit_if_enabled(parallel=True)
def main_solver_loop(par, sol, do_print = False):

    human_capital_unemp, hours_unemp, e_unemployed = 0.0, 0.0, 0.0

    # sol_a = sol.a
    # sol_ex = sol.ex
    sol_c = sol.c
    sol_h = sol.h
    sol_V = sol.V
    sol_c_given_m = sol.c_given_m

    efter = par.efter
    
    for t in range(par.T - 1, -1, -1):
        if do_print:
            print(f"We are in t = {t}")

        n_ret = min(par.last_retirement + 1, t + 1)
        for retirement_age_idx in range(n_ret):
            retirement_age = retirement_age_idx

            if t > par.last_retirement:
                e_grid = [par.ret]
            elif t >= par.retirement_age:
                e_grid = [par.emp, par.ret]
            else:
                e_grid = [par.unemp, par.emp, par.ret]

            for employed in e_grid:

                Na = len(par.a_grid)
                Ns = len(par.s_grid)
                Nk = len(par.k_grid[t])
                N_total = Na * Ns * Nk

                if t <= retirement_age:
                    if employed == par.emp or employed == par.unemp:
                        sol_EV = precompute_EV_next(par, sol_V, retirement_age_idx, employed, t)
                    else:
                        pass

                    if employed == par.emp:
                        if par.flexible_hours == "NVFI":

                            income, retirement_contribution = final_income_and_retirement_contri(par, par.a_grid[-1], par.s_grid[-1], par.k_grid[t][-1], par.h_max, employed, retirement_age, efter, t)

                            a_max_given_h = par.a_grid[-1] + income
                            s_max_given_h = par.s_grid[-1] + retirement_contribution
                            k_max_given_h = par.k_grid[t][-1] + (par.h_max / (1-par.delta))

                            par.a_m_grid[t, :, retirement_age_idx, employed] = nonlinspace(par.a_min, a_max_given_h, par.N_a, par.a_sp)
                            par.s_m_grid[t, :, retirement_age_idx, employed] = nonlinspace(par.s_min, s_max_given_h, par.N_s, par.s_sp)
                            par.k_m_grid[t, :, retirement_age_idx, employed] = nonlinspace(par.k_min, k_max_given_h, par.N_k, par.k_sp)

                            for flat in prange(N_total):
                                a_idx = flat // (Ns * Nk)
                                rem = flat - a_idx * (Ns * Nk)
                                s_idx = rem // Nk
                                k_idx = rem - s_idx * Nk

                                a_m = par.a_m_grid[t, a_idx, retirement_age_idx, employed]
                                s_m = par.s_m_grid[t, s_idx, retirement_age_idx, employed]
                                k_m = par.k_m_grid[t, k_idx, retirement_age_idx, employed]

                                idx = (t, a_idx, s_idx, k_idx, retirement_age_idx, employed)
                                idx_next = (t+1, a_idx, s_idx, k_idx, retirement_age_idx, employed)
                                
                                sol_c_given_m[idx] = optimizer_with_start(
                                    obj_consumption_given_m,
                                    optimizer_guess(sol_c_given_m[idx_next], par.c_min, a_m),
                                    par.c_min,
                                    a_m,
                                    par.speed,
                                    args=(par, sol_EV, a_m, s_m, k_m, t)
                                ) 
                            else:
                                pass
                    

                for flat in prange(N_total):
                    a_idx = flat // (Ns * Nk)
                    rem = flat - a_idx * (Ns * Nk)
                    s_idx = rem // Nk
                    k_idx = rem - s_idx * Nk

                    assets = par.a_grid[a_idx]
                    savings = par.s_grid[s_idx]
                    human_capital = par.k_grid[t][k_idx]

                    idx = (t, a_idx, s_idx, k_idx, retirement_age_idx, employed)
                    idx_next = (t+1, a_idx, s_idx, k_idx, retirement_age_idx, employed)
                    idx_ret = (t, a_idx, s_idx, slice(None), retirement_age_idx, employed)

                    if t == par.T - 1: # Last period
                        if k_idx == 0: # No capital
                            income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital_unemp, hours_unemp, employed, retirement_age, efter, t)

                            sol_c[idx_ret] = calculate_last_period_consumption(par, assets, savings, employed, retirement_age, t)
                            # sol_a[idx_ret] = (1+par.r_a)*(cash_on_hand - sol_c[idx])
                            # sol_ex[idx_ret] = e_unemployed
                            sol_h[idx_ret] = hours_unemp
                            sol_V[idx_ret] = value_last_period(par, sol_c[idx], assets, savings, employed, retirement_age, t)

                            if math.isnan(sol_V[idx]):
                                print("val is nan in first", idx, sol_V[idx])
                        else:
                            pass

                    elif t > retirement_age: # After retirement age, with "ratepension"
                        if t>= par.retirement_age:
                            if k_idx == 0: # No capital
                                income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital_unemp, hours_unemp, employed, retirement_age, efter, t)

                                bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital_unemp, employed, retirement_age, efter, income, t)

                                c_star = optimizer_with_start(
                                    obj_consumption_after_retirement,
                                    optimizer_guess(sol_c[idx_next], bc_min, bc_max), 
                                    bc_min,
                                    bc_max,
                                    par.speed,
                                    args=(par, sol_V, assets, savings, employed, retirement_age, efter, income, t)
                                )

                                sol_c[idx_ret] = c_star
                                # sol_a[idx_ret] = (1+par.r_a)*(cash_on_hand - sol_c[idx])
                                # sol_ex[idx_ret] = e_unemployed
                                sol_h[idx_ret] = hours_unemp
                                sol_V[idx_ret] = value_function_after_retirement(par, sol_V, c_star, assets, savings, employed, retirement_age, efter, income, t)

                                if math.isnan(sol_V[idx]):
                                    print("val is nan in second", idx, sol_V[idx])
                            else:
                                pass
                        else:
                            if k_idx == 0: # No capital
                                income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital_unemp, hours_unemp, employed, retirement_age, efter, t)

                                bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital_unemp, employed, retirement_age, efter, income, t)

                                c_star = optimizer_with_start(
                                    obj_consumption_after_retirement,
                                    optimizer_guess(sol_c[idx_next], bc_min, bc_max),
                                    bc_min,
                                    bc_max,
                                    par.speed,
                                    args=(par, sol_V, assets, savings, employed, retirement_age, efter, income, t)
                                )

                                sol_c[idx_ret] = c_star
                                # sol_a[idx_ret] = (1+par.r_a)*(cash_on_hand - sol_c[idx])
                                # sol_ex[idx_ret] = e_unemployed
                                sol_h[idx_ret] = hours_unemp
                                sol_V[idx_ret] = value_function_after_retirement(par, sol_V, c_star, assets, savings, employed, retirement_age, efter, income, t)

                                if math.isnan(sol_V[idx]):
                                    print("val is nan in second", idx, sol_V[idx])
                            else:
                                pass

                    elif t == retirement_age and t >= par.first_retirement:
                        if employed == par.unemp: # Forced unemployment
                            income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital, hours_unemp, employed, retirement_age, efter, t)

                            bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital, employed, retirement_age, efter, income, t)

                            c_star_u = optimizer_with_start(
                                obj_consumption_after_retirement,
                                optimizer_guess(sol_c[idx_next], bc_min, bc_max),
                                bc_min,
                                bc_max,
                                par.speed,
                                args=(par, sol_V, assets, savings, employed, retirement_age, efter, income, t)
                            )

                            sol_V[idx] = value_function_after_retirement(par, sol_V, c_star_u, assets, savings, employed, retirement_age, efter, income, t)
                            sol_c[idx]  = c_star_u
                            # sol_a[idx] = (1+par.r_a)*(cash_on_hand_un - sol_c[idx])
                            # sol_ex[idx] = e_unemployed
                            sol_h[idx]  = hours_unemp

                            if math.isnan(sol_V[idx]):
                                print("val is nan in third", idx, sol_V[idx])

                        elif employed == par.emp: # Can choose between employment and unemployment
                            if t >= par.retirement_age:

                                if par.flexible_hours == "VFI":
                                    h_star = optimize_outer(
                                        obj_hours,       
                                        par.h_min,
                                        par.h_max,
                                        args=(par, sol_V, sol_EV, assets, savings, human_capital, employed, retirement_age, efter, t)
                                    )
                                elif par.flexible_hours == "NVFI":
                                    h_star = optimizer_with_start(
                                        obj_hours_NVFI,
                                        optimizer_guess(sol_h[idx_next], par.h_min, par.h_max),
                                        par.h_min,
                                        par.h_max,
                                        par.speed,
                                        args=(par, sol_c_given_m, sol_EV, assets, savings, human_capital, employed, retirement_age, efter, t)
                                    )
                                else:
                                    h_star = par.hours_mean

                                income, retirement_contribution = final_income_and_retirement_contri(par, assets, savings, human_capital, h_star, employed, retirement_age, efter, t)

                                bc_min, bc_max = budget_constraint(par, h_star, assets, savings, human_capital, employed, retirement_age, efter, income, t)

                                c_star = optimizer_with_start(
                                    obj_consumption,
                                    optimizer_guess(sol_c[idx_next], bc_min, bc_max),
                                    bc_min,
                                    bc_max,
                                    par.speed,
                                    args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, employed, retirement_age, efter, income, retirement_contribution, t)
                                )

                                val = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, employed, retirement_age, efter, income, retirement_contribution, t)
                                
                                sol_V[idx] = val
                                sol_h[idx]  = h_star
                                sol_c[idx] = c_star
                                # sol_a[idx] = (1+par.r_a)*(cash_on_hand - sol_c[idx])

                                if math.isnan(sol_V[idx]):
                                    print("val is nan in fourth", idx, sol_V[idx])

                                # if sol_V[idx_unemployed] >= val:
                                #     sol_ex[idx] = e_unemployed
                                # else:
                                #     sol_ex[idx] = employed

                            else:
                                # h_star = optimize_outer(
                                #     obj_hours,       
                                #     par.h_min,
                                #     par.h_max,
                                #     args=(par, sol_V, sol_EV, assets, savings, human_capital, employed, retirement_age, efter, t),
                                #     tol=par.opt_tol
                                # )

    
                                if par.flexible_hours == "VFI":
                                    h_star = optimize_outer(
                                        obj_hours,       
                                        par.h_min,
                                        par.h_max,
                                        args=(par, sol_V, sol_EV, assets, savings, human_capital, employed, retirement_age, efter, t)
                                    )
                                elif par.flexible_hours == "NVFI":
                                    h_star = optimizer_with_start(
                                        obj_hours_NVFI,
                                        optimizer_guess(sol_h[idx_next], par.h_min, par.h_max),
                                        par.h_min,
                                        par.h_max,
                                        par.speed,
                                        args=(par, sol_c_given_m, sol_EV, assets, savings, human_capital, employed, retirement_age, efter, t)
                                    )
                                else:
                                    h_star = par.hours_mean

                                income, retirement_contribution = final_income_and_retirement_contri(par, assets, savings, human_capital, h_star, employed, retirement_age, efter, t)

                                bc_min, bc_max = budget_constraint(par, h_star, assets, savings, human_capital, employed, retirement_age, efter, income, t)

                                c_star = optimizer_with_start(
                                    obj_consumption,
                                    optimizer_guess(sol_c[idx_next], bc_min, bc_max),
                                    bc_min,
                                    bc_max,
                                    par.speed,
                                    args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, employed, retirement_age, efter, income, retirement_contribution, t)
                                )

                                val = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, employed, retirement_age, efter, income, retirement_contribution, t)

                                cash_on_hand = assets + income
                                sol_V[idx] = val
                                sol_h[idx]  = h_star
                                sol_c[idx] = c_star
                                # sol_a[idx] = (1+par.r_a)*(cash_on_hand - sol_c[idx])

                                if math.isnan(sol_V[idx]):
                                    print("val is nan in fourth", idx, sol_V[idx])

                                # if sol_V[idx_unemployed] >= val:
                                #     sol_ex[idx] = e_unemployed
                                # else:
                                #     sol_ex[idx] = employed

                        else: # Forced unemployment
                            if k_idx == 0: # No capital

                                income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital_unemp, hours_unemp, employed, retirement_age, efter, t)

                                bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital_unemp, employed, retirement_age, efter, income, t)

                                c_star_u = optimizer_with_start(
                                    obj_consumption_after_retirement,
                                    optimizer_guess(sol_c[idx_next], bc_min, bc_max),
                                    bc_min,
                                    bc_max,
                                    par.speed,
                                    args=(par, sol_V, assets, savings, employed, retirement_age, efter, income, t)
                                )

                                sol_V[idx_ret] = value_function_after_retirement(par, sol_V, c_star_u, assets, savings, employed, retirement_age, efter, income, t)
                                sol_c[idx_ret]  = c_star_u
                                # sol_a[idx_ret] = (1+par.r_a)*(cash_on_hand_un - sol_c[idx_ret])
                                # sol_ex[idx_ret] = e_unemployed
                                sol_h[idx_ret]  = hours_unemp

                                if math.isnan(sol_V[idx]):
                                    print("val is nan fifth", idx_ret, sol_V[idx])
                            else:
                                pass

                    else:
                        if employed == par.unemp: # Forced unemployment
                            income, retirement_contribution = final_income_and_retirement_contri(par, assets, savings, human_capital, hours_unemp, employed, retirement_age, efter, t)

                            bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital, employed, retirement_age, efter, income, t)
                            
                            c_star_u = optimizer_with_start(
                                obj_consumption,
                                optimizer_guess(sol_c[idx_next], bc_min, bc_max),
                                bc_min,
                                bc_max,
                                par.speed,
                                args=(par, sol_V, sol_EV, hours_unemp, assets, savings, human_capital, employed, retirement_age, efter, income, retirement_contribution, t)
                            )

                            sol_V[idx] = value_function(par, sol_V, sol_EV, c_star_u, hours_unemp, assets, savings, human_capital, employed, retirement_age, efter, income, retirement_contribution, t) 
                            sol_c[idx]  = c_star_u
                            # sol_a[idx] = (1+par.r_a)*(cash_on_hand_un - sol_c[idx])
                            # sol_ex[idx] = e_unemployed
                            sol_h[idx]  = hours_unemp

                            if math.isnan(sol_V[idx]):
                                print("val is nan in sixth", idx_ret, sol_V[idx])

                        elif employed == par.emp: # Can choose between employment and unemployment
                            # h_star = optimize_outer(
                            #     obj_hours,       
                            #     par.h_min,
                            #     par.h_max,
                            #     args=(par, sol_V, sol_EV, assets, savings, human_capital, employed, retirement_age, efter, t),
                            #     tol=par.opt_tol
                            # )
                        
                            if par.flexible_hours == "VFI":
                                h_star = optimize_outer(
                                    obj_hours,       
                                    par.h_min,
                                    par.h_max,
                                    args=(par, sol_V, sol_EV, assets, savings, human_capital, employed, retirement_age, efter, t)
                                )
                            elif par.flexible_hours == "NVFI":
                                h_star = optimizer_with_start(
                                    obj_hours_NVFI,
                                    optimizer_guess(sol_h[idx_next], par.h_min, par.h_max),
                                    par.h_min,
                                    par.h_max,
                                    par.speed,
                                    args=(par, sol_c_given_m, sol_EV, assets, savings, human_capital, employed, retirement_age, efter, t)
                                )
                            else:
                                h_star = par.hours_mean

                            income, retirement_contribution = final_income_and_retirement_contri(par, assets, savings, human_capital, h_star, employed, retirement_age, efter, t)

                            bc_min, bc_max = budget_constraint(par, h_star, assets, savings, human_capital, employed, retirement_age, efter, income, t)

                            c_star = optimizer_with_start(
                                obj_consumption,
                                optimizer_guess(sol_c[idx_next], bc_min, bc_max),
                                bc_min,
                                bc_max,
                                par.speed,
                                args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, employed, retirement_age, efter, income, retirement_contribution, t)
                            )

                            val = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, employed, retirement_age, efter, income, retirement_contribution, t)
                            # cash_on_hand = assets + income
                            sol_V[idx] = val
                            sol_c[idx] = c_star
                            sol_h[idx] = h_star
                            # sol_a[idx] = (1+par.r_a)*(cash_on_hand - sol_c[idx])

                            if math.isnan(sol_V[idx]):
                                print("val is nan in seventh", idx, sol_V[idx])

                            # if sol_V[idx_unemployed] >= val:
                            #     sol_ex[idx] = e_unemployed
                            # else:
                            #     sol_ex[idx] = employed
                        

                        else: # Forced unemployment

                            income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital_unemp, hours_unemp, employed, retirement_age, efter, t)

                            bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital_unemp, employed, retirement_age, efter, income, t)

                            c_star_u = optimizer_with_start(
                                obj_consumption_after_retirement,
                                optimizer_guess(sol_c[idx_next], bc_min, bc_max),
                                bc_min,
                                bc_max,
                                par.speed,
                                args=(par, sol_V, assets, savings, employed, retirement_age, efter, income, t)
                            )

                            sol_V[idx_ret] = value_function_after_retirement(par, sol_V, c_star_u, assets, savings, employed, retirement_age, efter, income, t)
                            sol_c[idx_ret]  = c_star_u
                            # sol_a[idx_ret] = (1+par.r_a)*(cash_on_hand_un - sol_c[idx_ret])
                            # sol_ex[idx_ret] = e_unemployed
                            sol_h[idx_ret]  = hours_unemp

                            if math.isnan(sol_V[idx]):
                                print("val is nan in eight", idx_ret, sol_V[idx])


    # return sol_c, sol_h, sol_ex, sol_V, sol_a
    return sol_c, sol_h, sol_V


# 6. simulation:
@jit_if_enabled(parallel=True)
def main_simulation_loop(par, sol, sim, do_print = False):
    '''Simulate the model: structure within each periode:
        1. technical variables
        2. interpolation of optimal consumption and hours
        3. income variables
        4. update of states'''
    sim_a = sim.a
    sim_s = sim.s
    sim_k = sim.k
    sim_c = sim.c
    sim_h = sim.h
    sim_e = sim.e

    sim_w = sim.w
    sim_ex = sim.ex
    sim_a_init = sim.a_init
    sim_s_init = sim.s_init
    sim_k_init = sim.k_init
    sim_income = sim.income
    sim_ret_flag    = sim.ret_flag
    sim_xi = sim.xi
    s_retirement = sim.s_retirement
    retirement_age = sim.retirement_age
    sim_s_lr_init = sim.s_lr_init
    sim_s_rp_init = sim.s_rp_init
    sim_chi_payment = sim.chi_payment
    sim_tax_rate = sim.tax_rate
    sim_income_before_tax_contrib = sim.income_before_tax_contrib
    sim_e_init = sim.e_init
    sim_e_exogenous = sim.e_state_exogenous
    efter = par.efter
    # sim_from_employed = sim.from_employed
    # sim_from_unemployed = sim.from_unemployed
    # sim_from_unemployed_to_only_early = sim.from_unemployed_to_only_early
    # sim_from_employed_to_unemployed = sim.from_employed_to_unemployed
    
    sim_s_retirement_contrib = sim.s_retirement_contrib
    
    # sol_ex = sol.ex
    sol_V = sol.V
    sol_c = sol.c
    sol_h = sol.h

    # i. initialize states
    sim_a[:,0] = sim_a_init[:]
    sim_s[:,0] = sim_s_init[:]
    sim_k[:,0] = sim_k_init[:]
    sim_e[:,0] = sim_e_init[:]


    for i in prange(par.simN):
        for t in range(par.simT):
            # ii. interpolate optimal consumption and hours
            if t < par.first_retirement:
            
                if t == 0:
                    retirement_age[i] = t
                    s_retirement[i] = sim_s[i,t]
                else:
                    if sim_e[i,t-1] == 2.0:
                        sim_e[i,t] = 2.0

                    else:
                        sim_e[i,t] = sim_e_exogenous[i,t]

                    if (sim_e[i,t] == 2.0 and sim_e[i,t-1] != 2) or sim_e[i,t] != 2.0:                    
                        retirement_age[i] = t
                        s_retirement[i] = sim_s[i,t] 


                if sim_e[i,t] == 2.0:
                    sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i])
                    sim_h[i,t] = 0.0
                    sim_ex[i,t] = 0.0
                    sim_ret_flag[i,t] = 0.0

                elif sim_e[i,t] == 1.0:
                    sol_v_unemp = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_V[t,:,:,:,int(retirement_age[i]), par.unemp], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    sol_v_emp = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_V[t,:,:,:,int(retirement_age[i]), par.emp], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    if sol_v_emp >= sol_v_unemp:
                        sim_ex[i,t] = 1
                    else:
                        sim_ex[i,t] = 0

                    if sim_ex[i,t] == 1.0:
                        sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_h[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_ret_flag[i,:] = 0.0 # hvis de kommer i arbejde igen, så skal de ikke have retirement flag

                    else:
                        sim_e[i,t] = 0
                        sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_h[i,t] = 0.0
                        sim_ret_flag[i,:] = 0.0 # glem alle tidligere
                        sim_ret_flag[i,t] = 1.0

                else:
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    sim_h[i,t] = 0.0
                    sim_ex[i,t] = 0.0
                    sim_ret_flag[i,t] = 0.0

                # 3. Income variables 
                # 3.1 final income and retirement payments 
                sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], efter, t) #(par, a, s, k, h, e, r, t)
                sim_s_lr_init[i], sim_s_rp_init[i] = calculate_retirement_payouts(par, sim_h[i,t], s_retirement[i], sim_e[i,t], retirement_age[i], t) # par, h, s, e, r, t

                if sim_a[i,t] +sim_income[i,t] - sim_c[i,t] < par.a_min:    
                    sim_c[i,t] = sim_a[i,t] +sim_income[i,t] - par.a_min

                if par.speed == "ROBUST":
                    # 3.2 labor income
                    sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], efter, sim_income[i,t], t)
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], efter, t) 
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], efter, t)
                else:
                    sim_w[i,t] = 0.0
                    sim_chi_payment[i,t] = 0.0
                    sim_income_before_tax_contrib[i,t] = 0.0 
                    sim_tax_rate[i,t] = 0.0

                # 4. Update of states
                # sim_a[i,t+1] = np.maximum(par.a_min, np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max))
                sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] + sim_s_retirement_contrib[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t], par.k_max[t])

                # if sim_a[i,t+1] < par.a_min:
                #     print("id", i, "time", t, "asspre", sim_a[i,t], "ass", sim_a[i,t+1], "inc", sim_income[i,t], "c", sim_c[i,t], "ex", sim_ex[i,t], "h", sim_h[i,t], "e", sim_e[i,t], "r", retirement_age[i])

            elif t < par.retirement_age:

                if sim_e[i,t-1] == 2.0:
                    sim_e[i,t] = 2.0

                elif sim_e[i,t-1] == 0.0:
                    if sim_e_exogenous[i,t] == 2:
                        sim_e[i,t] = 2.0
                    else:
                        sim_e[i,t] = 0.0
                
                else:
                    sim_e[i,t] = sim_e_exogenous[i,t]

                if ((sim_e[i,t] == 2.0 or sim_e[i,t] == 0) and sim_e[i,t-1] == 1.0) or sim_e[i,t] == 1:                    
                    retirement_age[i] = t
                    s_retirement[i] = sim_s[i,t] 

                if sim_e[i,t] == 2.0:
                    sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i])
                    sim_h[i,t] = 0.0
                    sim_ex[i,t] = 0.0
                    sim_ret_flag[i,t] = 0.0

                elif sim_e[i,t] == 1.0:
                    sol_v_unemp = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_V[t,:,:,:,int(retirement_age[i]), par.unemp], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    sol_v_emp = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_V[t,:,:,:,int(retirement_age[i]), par.emp], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    if sol_v_emp >= sol_v_unemp:
                        sim_ex[i,t] = 1
                    else:
                        sim_ex[i,t] = 0

                    if sim_ex[i,t] == 1.0:
                        sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_h[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_ret_flag[i,:] = 0.0 # hvis de kommer i arbejde igen, så skal de ikke have retirement flag

                    else:
                        sim_e[i,t] = 0
                        sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_h[i,t] = 0.0
                        sim_ret_flag[i,:] = 0.0 # glem alle tidligere
                        sim_ret_flag[i,t] = 1.0

                else:
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    sim_h[i,t] = 0.0
                    sim_ex[i,t] = 0.0
                    sim_ret_flag[i,t] = 0.0

                sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], efter, t) #(par, a, s, k, h, e, r, t)
                sim_s_lr_init[i], sim_s_rp_init[i] = calculate_retirement_payouts(par, sim_h[i,t], s_retirement[i], sim_e[i,t], retirement_age[i], t) # par, h, s, e, r, t

                if sim_a[i,t] +sim_income[i,t] - sim_c[i,t] < par.a_min:    
                    sim_c[i,t] = sim_a[i,t] +sim_income[i,t] - par.a_min

                if par.speed == "ROBUST":
                    sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], efter, sim_income[i,t], t)
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], efter, t) 
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], efter, t)
                else:
                    sim_w[i,t] = 0.0
                    sim_chi_payment[i,t] = 0.0
                    sim_income_before_tax_contrib[i,t] = 0.0 
                    sim_tax_rate[i,t] = 0.0

                # 4. Update of states
                # sim_a[i,t+1] = np.maximum(par.a_min, np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max))
                sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] + sim_s_retirement_contrib[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t], par.k_max[t])


            elif t <= par.last_retirement:

                if sim_e[i,t-1] == 2.0 or sim_e[i,t-1] == 0.0:
                    sim_e[i,t] = 2.0

                else:
                    if sim_e_exogenous[i,t] == 1:
                        sim_e[i,t] = 1.0
                    else:
                        sim_e[i,t] = 2.0


                if ((sim_e[i,t] == 2.0 or sim_e[i,t] == 0) and sim_e[i,t-1] == 1.0) or sim_e[i,t] == 1:
                    retirement_age[i] = t
                    s_retirement[i] = sim_s[i,t] 

                if sim_e[i,t] == 2.0:
                    sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i])
                    sim_h[i,t] = 0.0
                    sim_ex[i,t] = 0.0
                    sim_ret_flag[i,t] = 0.0

                elif sim_e[i,t] == 1.0:
                    sol_v_unemp = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_V[t,:,:,:,int(retirement_age[i]), par.ret], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    sol_v_emp = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_V[t,:,:,:,int(retirement_age[i]), par.emp], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    if sol_v_emp >= sol_v_unemp:
                        sim_ex[i,t] = 1
                    else:
                        sim_ex[i,t] = 0

                    if sim_ex[i,t] == 1.0:
                        sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_h[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_ret_flag[i,:] = 0.0 # hvis de kommer i arbejde igen, så skal de ikke have retirement flag

                    else:
                        sim_e[i,t] = 2
                        sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_h[i,t] = 0.0
                        sim_ret_flag[i,:] = 0.0 # glem alle tidligere
                        sim_ret_flag[i,t] = 1.0

                else:
                    sim_e[i,t] = 2
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    sim_h[i,t] = 0.0
                    sim_ex[i,t] = 0.0
                    sim_ret_flag[i,t] = 0.0

                sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], efter, t) #(par, a, s, k, h, e, r, t)
                sim_s_lr_init[i], sim_s_rp_init[i] = calculate_retirement_payouts(par, sim_h[i,t], s_retirement[i], sim_e[i,t], retirement_age[i], t) # par, h, s, e, r, t

                if sim_a[i,t] +sim_income[i,t] - sim_c[i,t] < par.a_min:    
                    sim_c[i,t] = sim_a[i,t] +sim_income[i,t] - par.a_min

                if par.speed == "ROBUST":
                    # 3.2 labor income
                    sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], efter, sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], efter, t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], efter, t)
                else:
                    sim_w[i,t] = 0.0
                    sim_chi_payment[i,t] = 0.0
                    sim_income_before_tax_contrib[i,t] = 0.0 
                    sim_tax_rate[i,t] = 0.0

                # 4. Update of states
                # sim_a[i,t+1] = np.maximum(par.a_min, np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max))
                sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] + sim_s_retirement_contrib[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t], par.k_max[t])

            elif t > par.last_retirement:
                sim_ex[i,t] = 0.0
                sim_e[i,t]  = 2.0
                sim_ret_flag[i,t] = 0.0

                # 1.1 retirement age
                # 2. Interpolation of choice variables
                sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i])
                sim_h[i,t] = 0.0

                # 3. Income variables
                sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], efter, t)

                if sim_a[i,t] +sim_income[i,t] - sim_c[i,t] < par.a_min:    
                    sim_c[i,t] = sim_a[i,t] +sim_income[i,t] - par.a_min

                # 3.1 retirement payments
                sim_s_lr_init[i], sim_s_rp_init[i] = calculate_retirement_payouts(par, sim_h[i,t], s_retirement[i], sim_e[i,t], retirement_age[i], t) # par, h, s, e, r, t

                if par.speed == "ROBUST":
                    # 3.2 labor income
                    sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], efter, sim_income[i,t], t)
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], efter, t) 
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], efter, t)
                else:
                    sim_w[i,t] = 0.0
                    sim_chi_payment[i,t] = 0.0
                    sim_income_before_tax_contrib[i,t] = 0.0 
                    sim_tax_rate[i,t] = 0.0

                if t < retirement_age[i] + par.m: 
                    # 4. Update of states
                    # sim_a[i,t+1] = np.maximum(par.a_min, np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max))
                    sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                    sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] + sim_s_retirement_contrib[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                    sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t])*sim_xi[i,t], par.k_max[t])

                elif par.T - 1 > t >= retirement_age[i] + par.m:
                    # 4. Update of states
                    # sim_a[i,t+1] = np.maximum(par.a_min, np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max))
                    sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                    sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] + sim_s_retirement_contrib[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                    sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t])*sim_xi[i,t], par.k_max[t])
                    

    # if par.speed == "ROBUST":
    return sim_a, sim_s, sim_k, sim_c, sim_h, sim_w, sim_ex, sim_e, sim_chi_payment, sim_tax_rate, sim_income_before_tax_contrib, s_retirement, retirement_age, sim_income, sim_ret_flag 

    # else:
    #     return sim_a, sim_s, sim_k, sim_c, sim_h, sim_ex, sim_e, s_retirement, retirement_age, sim_income, sim_ret_flag 


