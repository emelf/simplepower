from scipy.optimize import OptimizeResult
from typing import Callable, Optional, Sequence
from copy import deepcopy
import numpy as np 
import pandas as pd 
from scipy.optimize import root

from ..common import PowerFlowResult, PQVD
from ..data_classes import GridDataClass
from ..component_models.base_models import BaseComponentModel 
from ..component_models.generator_models import BasePQGenerator, BasePVGenerator
from ..component_models.load_PQ_models import BasePQLoad

class GridModel: 
    def __init__(self, grid_data: GridDataClass): 
        self.md = grid_data 
        self.y_bus = self.md.get_Y_bus() 
        self.y_lines = self.md.get_Y_lines() 
        self.P_mask, self.Q_mask, self.V_mask, self.delta_mask = self.md.get_PQVd_mask()  

    def _do_pf(self, V_vals, delta_vals, y_bus): 
        # Convert to a vector of complex voltages 
        V_vec = V_vals*np.cos(delta_vals) + V_vals*np.sin(delta_vals)*1j
        # V_mat = np.eye(len(V_vec))*V_vec  
        S_conj = V_vec.conj() * (y_bus @ V_vec)
        P_calc = S_conj.real 
        Q_calc = -S_conj.imag 
        return P_calc, Q_calc

    def _setup_pf(self, ts: int) -> Callable[[np.ndarray], np.ndarray]:
        """ 
        Creates and returns the correct functions for doing a power flow calculation. 
        
        Attributes 
        -----------
        """ 

        V_vals, delta_vals = self.md._get_V_delta_vals(ts)

        def pf_eqs(X): 
            _delta_vals = delta_vals.copy()
            _V_vals = V_vals.copy()
            
            _delta_vals[self.delta_mask] = X[:self.md.N_delta] # Convention that X = [delta, V]
            _V_vals[self.V_mask] = X[self.md.N_delta:]

            P_calc, Q_calc = self._do_pf(_V_vals, _delta_vals, self.y_bus)
            pqvd_res = PQVD(P_calc, Q_calc, _V_vals, _delta_vals, self.md.S_base_mva, self.md.V_base_kV)
            P_vals, Q_vals = self.md._get_PQ_vals(pqvd_res, ts) 

            P_root = (P_calc - P_vals)[self.P_mask]
            Q_root = (Q_calc - Q_vals)[self.Q_mask]

            S_root = np.zeros(len(X)) # Do this initialization to be numba compatible 
            S_root[:len(P_root)] = P_root 
            S_root[len(P_root):] = Q_root
            return S_root 
        
        return pf_eqs 
    
    def _calc_pf(self, ts: int, method: str) -> PowerFlowResult: 
        X0 = np.zeros(len(self.delta_mask) + len(self.V_mask))
        X0[self.md.N_delta:] = 1.0 # flat voltage start
        pf_eqs = self._setup_pf(ts) 
        sol = root(pf_eqs, X0, tol=1e-8, method=method)
        return sol
    
    def _get_pf_sol(self, sol: OptimizeResult, ts: int): 
        V_vals, delta_vals = self.md._get_V_delta_vals(ts)
        delta_vals[self.delta_mask] = sol.x[:self.md.N_delta]
        V_vals[self.V_mask] = sol.x[self.md.N_delta:]
        P_calc, Q_calc = self._do_pf(V_vals, delta_vals, self.y_bus) 
        return PowerFlowResult(P_calc, Q_calc, V_vals, delta_vals, self.md.S_base_mva, sol)
        
    def powerflow(self, time_index: Optional[Sequence[int]]=None, method="hybr") -> PowerFlowResult: 
        """ 
        If ts (time step) is None, performs powerflow calculation on all available data. 
        
        Returns 
        ---------
        PowerFlowResult 
        """
        if time_index is None: 
            time_index = 0
        sol = self._calc_pf(time_index, method)
        if not sol.success: 
            print(sol)
        sol = self._get_pf_sol(sol, time_index) 
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
    
    def _get_from_models(self, pqvd: PQVD, time_index: int): 
        P_vals, Q_vals = self.md._get_PQ_vals(pqvd, time_index)
        V_vals, delta_vals = self.md._get_V_delta_vals(pqvd, time_index)
        return P_vals, Q_vals 
        
    def convert_PV_to_PQ_grid(self, ts): 
        grid_data_PQ = deepcopy(self.md) 
        grid_model = GridModel(grid_data_PQ)
        pf_res = grid_model.powerflow(ts)
        # Adding a load for each generator after the power flow*
        N_PQ_gens= len(grid_data_PQ._grid_static_gens)
        idx = 0
        for _, gen in grid_data_PQ._grid_gens.iterrows(): 
            if gen["is_slack"] != 1:
                new_data = {"name": gen["name"], "S_rated_mva": gen["S_rated_mva"], 
                            "p_set_mw": gen["p_set_mw"], "q_set_mvar": pf_res.Q_calc[gen["bus_idx"]], 
                            "bus_idx": gen["bus_idx"]}
                grid_data_PQ._grid_static_gens.loc[N_PQ_gens+idx] = pd.Series(new_data) 
                idx += 1 
                
        # Code for removing all generators except the slack 
        idx_slack = np.argmax(grid_data_PQ._grid_gens["is_slack"] == 1)
        N_gens = len(grid_data_PQ._grid_gens)
        gen_idx = [i for i in range(N_gens) if i != idx_slack]
        grid_data_PQ._grid_gens.drop(index=gen_idx, inplace=True)
        grid_data_PQ._define_PQV_models([])
        grid_data_PQ._give_added_models(self.md.added_models)
        return grid_data_PQ

                
