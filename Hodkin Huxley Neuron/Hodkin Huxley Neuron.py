import numpy as np
import sympy as sp
import scipy.integrate as spi
import Hodkin_Huxley_Neuron_Functions as hh



def hodgkin_huxley_ode(y, t):
    V_m, m, h, n = y
    dy_dt = [hh.dV_m_dt(C_m, hh.I_Na(g_Na, m, h, V_m, E_Na), hh.I_K(g_K, n, V_m, E_K), hh.I_L(g_L, V_m, E_L), hh.I_ext(t)),
             hh.dm_dt(hh.alpha_m(V_m), hh.beta_m(V_m), m),
             hh.dh_dt(hh.alpha_h(V_m), hh.beta_h(V_m), h),
             hh.dn_dt(hh.alpha_n(V_m), hh.beta_n(V_m), n)]
    # Constants
    C_m = 1.0
    g_Na = 120.0
    g_K = 36.0
    g_L = 0.3
    E_Na = 115.0
    E_K = -12.0
    E_L = 10.6

    return dy_dt

