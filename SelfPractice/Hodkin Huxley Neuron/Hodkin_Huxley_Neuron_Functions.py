import numpy as np
import sympy as sp
import scipy.integrate as spi


def I_ext(t):
    I_ext = 0
    return I_ext


def dV_m_dt(C_m, I_Na, I_K, I_L, I_ext):
    dV_m_dt = (I_ext - I_Na - I_K - I_L) / C_m
    return dV_m_dt


def I_Na(g_Na, m, h, V_m, E_Na):
    I_Na = g_Na * (m**3) * h * (V_m - E_Na)
    return I_Na


def I_K(g_K, n, V_m, E_K):
    I_K = g_K * (n**4) * (V_m - E_K)
    return I_K


def I_L(g_L, V_m, E_L):
    I_L = g_L * (V_m - E_L)
    return I_L


def dm_dt(alpha_m, beta_m, m):
    dm_dt = alpha_m * (1 - m) - beta_m * m
    return dm_dt


def alpha_m(V_m):
    alpha_m = 0.1 * (V_m + 40) / (1 - np.exp(-(V_m + 40) / 10))
    return alpha_m 


def beta_m(V_m):
    beta_m = 4 * np.exp(-(V_m + 65) / 18)
    return beta_m


def dh_dt(alpha_h, beta_h, h):
    dh_dt = alpha_h * (1 - h) - beta_h * h
    return dh_dt

def alpha_h(V_m):
    alpha_h = 0.07 * np.exp(-(V_m + 65) / 20)
    return alpha_h

def beta_h(V_m):
    beta_h = 1 / (1 + np.exp(-(V_m + 35) / 10))
    return beta_h

def dn_dt(alpha_n, beta_n, n):
    dn_dt = alpha_n * (1 - n) - beta_n * n
    return dn_dt

def alpha_n(V_m):
    alpha_n = 0.01 * (V_m + 55) / (1 - np.exp(-(V_m + 55) / 10))
    return alpha_n

def beta_n(V_m):
    beta_n = 0.125 * np.exp(-(V_m + 65) / 80)
    return beta_n