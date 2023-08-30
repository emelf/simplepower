from scipy.optimize import OptimizeResult
from typing import Callable
from numpy.typing import ArrayLike

import numpy as np 
import pandas as pd 
from scipy.optimize import root

from simplepower.Utils import PowerFlowResult
from simplepower.Dataclasses import GridDataClass

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
    
    def _get_I_mat(self, V_vec):         
        V_mat = np.zeros((len(V_vec), len(V_vec)), dtype=np.complex64)
        for idx, V in enumerate(V_vec):
            V_mat[idx] = -V_vec
            V_mat[idx] += V
            V_mat[idx, idx] = V   

        I_mat = V_mat * self.y_lines
        return I_mat 
    
    def _get_S_mat(self, V_vals, d_vals): 
        V_vec = V_vals*np.cos(d_vals) + V_vals*np.sin(d_vals)*1j
        V_mat = np.array([V*np.ones(len(V_vec), dtype=np.complex64) for V in V_vec])
        I_mat = self._get_I_mat(V_vec)
        S_mat = I_mat * V_mat.conj()
        return S_mat.conj()
    
    def get_line_violations(self, V_vals, d_vals): 
        V_vec = V_vals*np.cos(d_vals) + V_vals*np.sin(d_vals)*1j
        I_mat = self._get_I_mat(V_vec)
        is_lim = (np.abs(I_mat) * (-np.eye(self.md.N_buses) + 1) > self.md.I_lims_pu)
        return pd.DataFrame(is_lim, columns=self.md.columns, index=self.md.indices)
                
