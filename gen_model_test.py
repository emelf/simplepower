from dataclasses import dataclass 
from typing import Optional
import numpy as np 
from scipy.optimize import root 

@dataclass 
class GenModelThirdOrderData: 
    S_nom_mva: float
    V_nom_kv: float 
    X_d_u: float 
    X_d_u_t: float 
    X_q_u: float
    X_l: float 
    R_a_nom: float 
    T_do_t: float
    H: float 
    D: float 
    
    def __post_init__(self): 
        self.X_ad_u = self.X_d_u - self.X_l  
        self.X_aq_u = self.X_q_u - self.X_l
        self._Z = np.array([[self.R_a_nom, self.X_q_u], [-self.X_d_u_t, self.R_a_nom]])
        self._Y = np.linalg.inv(self._Z)
        self.N_states = 3
        
        
class GenModelThirdOrder: 
    def __init__(self, gen_data: GenModelThirdOrderData): 
        """Model description: 
        X = [delta, D_omega, E_q_t], 
        y = [P_e, Q_e]
        u = [E_f, P_m]"""
        self.data = gen_data 
    
    def init_state(self, P_g_pu, Q_g_pu, V_g_pu, delta_t):
        """Initializes the dynamic model from the power flow solution."""
        # This init function must find the following quantities: 
        
        # [delta, D_omega, E_q_t, P_e, I_d, I_q, X_d, X_q, k_d, k_q, E_f, P_m]
        I_a = np.sqrt(P_g_pu**2 + Q_g_pu**2)/V_g_pu
        phi = np.arctan(Q_g_pu/P_g_pu)
        I_d_init = -I_a*np.sin(phi)
        I_q_init = I_a*np.cos(phi)
        x0 = np.array([0, 0, delta_t, P_g_pu, I_d_init, I_q_init, self.X_d_u, self.X_q_u, 
                       1.0, 1.0, delta_t, P_g_pu])
        sol = root(self._init_from_pf_objective, x0, args=(P_g_pu, Q_g_pu, delta_t, V_g_pu))
        delta, D_omega, E_q_t, P_e, I_d, I_q, X_d, X_q, k_d, k_q, E_f, P_m = sol.x
        X0 = np.array([delta, D_omega, E_q_t])
        y0 = np.array([P_e, I_d, I_q, P_g_pu, Q_g_pu, X_d, X_q, k_d, k_q])
        u0 = np.array([E_f, P_m, Q_g_pu, delta_t])
        return X0, y0, u0

    def _dx_dt(self, X, y, u): 
        """Model description: 
        X = [delta, D_omega, E_q_t], 
        y = [P_e, Q_e]
        u = [E_f, P_m]"""
        delta, D_omega, E_q_t = X 
        P_e, V_d, V_q, I_d, I_q = y 
        E_f, P_m = u
        
        ddelta_dt = D_omega
        dD_omega_dt = (P_m - P_e - self.data.D*D_omega)/(2.0*self.data.H) 
        dE_q_t_dt = (E_f - E_q_t + I_d*(self.data.X_d_u - self.data.X_d_u_t))/self.data.T_do_t
        return np.array([ddelta_dt, dD_omega_dt, dE_q_t_dt], dtype=np.float64)

    def _T_f(self, X):
        delta, _, _, _ = X 
        return np.array([[-np.sin(delta), np.cos(delta)], [np.cos(delta), np.sin(delta)]])
    
    def _init_from_pf_objective(self, x, P_g, Q_g, V_t, delta_t): 
        delta, D_omega, E_q_t, P_e, I_d, I_q, X_d, X_q, k_d, k_q, E_f, P_m = x
        V_a, V_b = np.array([V_t*np.cos(delta_t), V_t*np.sin(delta_t)])
        V_dq = self._T_f(delta) @ np.array([V_a, V_b])

        # First assume the state diff equations are 0 
        f1 = D_omega 
        f2 = P_m - P_e - self.data.D*D_omega
        f3 = E_f - E_q_t + I_d*(X_d - self.X_d_t)

        # Then find the root of the algebraic equations
        f4 = P_e - (E_q_t*I_q + (self.data.X_d_u_t - self.data.X_q_u)*I_d*I_q)
        f5 = P_g - (V_dq[0]*I_d + V_dq[1]*I_q)
        f6 = Q_g - (-V_dq[1]*I_d + V_dq[0]*I_q)
        f7, f8 = np.array([I_d, I_q]) - (self.data._Y@(np.array([0, E_q_t]) - V_dq))

        return np.array([f1,f2,f3,f4,f5,f6,f7,f8])
    
    def get_alg_vars(self, X, u): 
        delta, D_omega, E_q_t = X
        E_f, P_m, V_t, delta_t = u
        V_a, V_b = np.array([V_t*np.cos(delta_t), V_t*np.sin(delta_t)])
        V_d, V_q = self._T_f(delta) @ np.array([V_a, V_b])
        I_d, I_q = self.Y @ (np.array([0, E_q_t]) - np.array([V_d, V_q]))

        P_g = V_d*I_d + V_q*I_q
        Q_g = -V_q*I_d + V_d*I_q
        P_e = E_q_t*I_q + (self.data.X_d_u_t - self.data.X_q_u)*I_d*I_q

        return np.array([P_e, I_d, I_q, P_g, Q_g])
        
    