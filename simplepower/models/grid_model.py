import numpy as np 
import pandas as pd 
from dataclasses import dataclass 
from scipy.optimize import root, OptimizeResult, minimize, LinearConstraint
from typing import Tuple, Sequence, Callable, Optional
from numpy.typing import ArrayLike
import cmath as cm

import os 
import inspect 
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils import PowerFlowResult
from models.grid_data_class import GridDataClass

class GridModel: 
    def __init__(self, grid_data: GridDataClass): 
        self.md = grid_data 
        self.y_bus = self.md.get_Y_bus() 
        self.y_lines = self.md.get_Y_lines() 
        # self.P_mask, self.Q_mask = self.md.get_PQ_mask() 
        # self.V_mask, self.delta_mask, _ = self.md.get_V_delta_mask() 
        self.P_mask, self.Q_mask, self.V_mask, self.delta_mask = self.md.get_PQVd_mask() 

    def _do_pf(self, V_vals, delta_vals, y_bus): 
        # Convert to a vector of complex voltages 
        V_vec = V_vals*np.cos(delta_vals) + V_vals*np.sin(delta_vals)*1j
        # V_mat = np.eye(len(V_vec))*V_vec  
        S_conj = V_vec.conj() * (y_bus @ V_vec)
        P_calc = S_conj.real 
        Q_calc = -S_conj.imag 
        return P_calc, Q_calc

    def _setup_pf(self) -> Callable[[np.ndarray], np.ndarray]:
        """ 
        Creates and returns the correct functions for doing a power flow calculation. 
        
        Attributes 
        -----------
        """ 
        V_vals, delta_vals = self.md.get_V_delta_vals()
        P_vals, Q_vals = self.md.get_PQ_vals() 

        def pf_eqs(X): 
            _delta_vals = delta_vals.copy()
            _V_vals = V_vals.copy() 
            
            _delta_vals[self.delta_mask] = X[:self.md.N_delta] # Convention that X = [delta, V]
            _V_vals[self.V_mask] = X[self.md.N_delta:]

            P_calc, Q_calc = self._do_pf(_V_vals, _delta_vals, self.y_bus)

            P_root = (P_calc - P_vals)[self.P_mask]
            Q_root = (Q_calc - Q_vals)[self.Q_mask]

            S_root = np.zeros(len(X)) # Do this initialization to be numba compatible 
            S_root[:len(P_root)] = P_root 
            S_root[len(P_root):] = Q_root
            return S_root 
        
        return pf_eqs 
    
    def _calc_pf(self, method) -> PowerFlowResult: 
        X0 = np.zeros(len(self.delta_mask) + len(self.V_mask))
        X0[self.md.N_delta:] = 1.0 # flat voltage start
        pf_eqs = self._setup_pf() 
        sol = root(pf_eqs, X0, tol=1e-8, method=method)
        return sol 
    
    def _get_pf_sol(self, sol: OptimizeResult): 
        V_vals, delta_vals = self.md.get_V_delta_vals()
        delta_vals[self.delta_mask] = sol.x[:self.md.N_delta]
        V_vals[self.V_mask] = sol.x[self.md.N_delta:]
        P_calc, Q_calc = self._do_pf(V_vals, delta_vals, self.y_bus) 
        return PowerFlowResult(P_calc, Q_calc, V_vals, delta_vals, self.md.S_base_mva, sol)
    
    def powerflow(self, method="hybr") -> PowerFlowResult: 
        """ 
        Does a power flow calculation based on the values specified in excel. 
        
        Returns 
        ---------
        PowerFlowResult 
        """
        sol = self._calc_pf(method)
        if not sol.success: 
            print(sol)
        sol = self._get_pf_sol(sol) 
        return sol 
    
    def _get_I_mat(self, V_vals, d_vals):         
        V_vec = V_vals*np.cos(d_vals) + V_vals*np.sin(d_vals)*1j
        V_mat = np.zeros((len(V_vec), len(V_vec)), dtype=np.complex64) * V_vec 
        for idx, V in enumerate(V_vec):
            V_mat[idx] = - V_vec
            V_mat[idx] += V
            V_mat[idx, idx] += V   

        I_mat = V_mat * self.y_lines
        return I_mat
                

class ORPDHandler: 
    def __init__(self, grid_model: GridModel, gen_control_idx: Sequence[int], 
                 V_min: Optional[ArrayLike]=0.95, 
                 V_max: Optional[ArrayLike]=1.05): 
        self.grid_model = grid_model 
        self.gen_control_idx = gen_control_idx
        self.ORPD_func = self.get_ORPD_func()
        self.V_min = V_min
        self.V_max = V_max 
        self.cons = LinearConstraint(np.eye(len(self.gen_control_idx)), lb=self.V_min, ub=self.V_max)

    def get_ORPD_func(self): 

        def ORPD(V_vals): 
            self.grid_model.md.change_V_gen(self.gen_control_idx, V_vals)
            sol = self.grid_model.powerflow()
            return sol.get_P_losses()
        return ORPD 
    
    def solve_ORPD(self) -> OptimizeResult: 
        sol = minimize(self.ORPD_func, x0=np.ones(len(self.gen_control_idx)), constraints=self.cons)
        return sol 
    
    def get_problem_dict(self, log_to="console") -> dict:
        problem_dict = {
        "fit_func": self.get_ORPD_func(),
        "lb": [0.9, ] * len(self.gen_control_idx),
        "ub": [1.1, ] * len(self.gen_control_idx),
        "minmax": "min",
        "log_to": log_to}
        return problem_dict
    
    def set_opt_v(self, sol: OptimizeResult): 
        self.grid_model.md.change_V_gen(self.gen_control_idx, sol.x)
