from typing import Optional, Sequence 
from numpy.typing import ArrayLike 
from scipy.optimize import OptimizeResult, NonlinearConstraint

import numpy as np 
from scipy.optimize import minimize
from copy import deepcopy

from simplepower.PowerFlowModels.GridModel import GridModel

class ORPDHandler: 
    def __init__(self, grid_model: GridModel, gen_control_idx: Sequence[int], 
                 V_min: Optional[ArrayLike]=None, 
                 V_max: Optional[ArrayLike]=None, 
                 I_lim_mat: Optional[ArrayLike]=None): 
        self.grid_model = grid_model 
        self.gen_control_idx = gen_control_idx
        self.ORPD_func = self.get_ORPD_func()
        if V_min is None:
            self.V_min = np.array([0.95]*self.grid_model.md.N_buses)
        elif type(V_min) is float: 
            self.V_min = np.ones(self.grid_model.md.N_buses, dtype=float)*V_min 
        else: 
            self.V_min = V_min
        if V_max is None:
            self.V_max = np.array([1.05]*self.grid_model.md.N_buses)
        elif type(V_max) is float: 
            self.V_max = np.ones(self.grid_model.md.N_buses, dtype=float)*V_max 
        else:
            self.V_max = V_max
        self._I_lim_mat = I_lim_mat
        self.I_min, self.I_max = self._define_I_lims(I_lim_mat) 
        self.lb = np.concatenate((self.V_min, self.I_min))
        self.ub = np.concatenate((self.V_max, self.I_max))

    def _define_I_lims(self, I_mat): 
        I_min = [] 
        I_max = []
        if I_mat is None: 
            pass 
        else:
            for i, I_row in enumerate(I_mat): 
                for j, I_val in enumerate(I_row): 
                    if j > i: 
                        if abs(I_val) > 1e-6:
                            I_min.append(-abs(I_val)) 
                            I_max.append(abs(I_val))
        return np.array(I_min), np.array(I_max) 

    def get_ORPD_func(self): 
        def ORPD(V_vals): 
            self.grid_model.md.change_V_gen(self.gen_control_idx, V_vals)
            sol = self.grid_model.powerflow()
            return sol.get_P_losses()
        return ORPD 
    
    def solve_ORPD(self, V0: Optional[Sequence[float]]=None) -> OptimizeResult: 
        if V0 is None: 
            V0 = np.ones(len(self.gen_control_idx))
        cons = (NonlinearConstraint(self.const, self.lb, self.ub))
        sol = minimize(self.ORPD_func, x0=V0, constraints=cons, jac='3-point')
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
    
    def const(self, X):
        grid_data = deepcopy(self.grid_model.md)
        grid_data.change_V_gen(self.gen_control_idx, X)
        grid_model = GridModel(grid_data) 
        sol = grid_model.powerflow()
        V_vals = sol.V_buses
        if self._I_lim_mat is not None: 
            delta_vals = sol.d_buses 
            V_vec = V_vals*np.cos(delta_vals) + V_vals*np.sin(delta_vals)*1j
            I_mat = grid_model._get_I_mat(V_vec) 
            _, I_vals = self._define_I_lims(I_mat)
            Y = np.concatenate((V_vals, I_vals))
        else: 
            Y = V_vals
        return Y
    
    def is_inside_const(self, V_vals): 
        const_vals = self.const(V_vals) 
        lb = all(const_vals > self.lb)
        ub = all(const_vals < self.ub) 
        return lb and ub 

