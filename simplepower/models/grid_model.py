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
from models.branch_model import LineDataClass, TrafoDataClass


class GridDataClass: 
    def __init__(self, filename: str, f_nom: float): 
        self._grid_buses = pd.read_excel(filename, sheet_name="busbars")
        self._grid_lines = pd.read_excel(filename, sheet_name="lines")
        self._grid_trafos  = pd.read_excel(filename, sheet_name="trafos")
        self._grid_loads = pd.read_excel(filename, sheet_name="loads")
        self._grid_gens  = pd.read_excel(filename, sheet_name="gens")
        
        self.S_base_mva = self._grid_gens["S_rated_mva"].sum()
        self.V_base_kV = self._grid_buses["v_nom_kv"].max()
        self.f_nom = f_nom
        self.N_buses = self._grid_buses.shape[0] 

        # Create all line models: 
        self._line_data = [] 
        for _, row in self._grid_lines.iterrows(): 
            v_base = row["v_nom_kv"]
            r = row["r_ohm_per_km"]
            x = row["x_ohm_per_km"] 
            c = row["c_uf_per_km"]
            i = row["from_bus_idx"]
            j = row["to_bus_idx"]
            length = row["length_km"]
            line_data = LineDataClass(self.S_base_mva, v_base, r, x, i, j, c, length, is_pu=bool(row["is_pu"]), f_nom=self.f_nom)

            if not bool(row["is_pu"]):
                line_data = line_data.convert_to_pu().change_base(self.S_base_mva, self.V_base_kV)
            else: 
                line_data = line_data.change_base(self.S_base_mva, self.V_base_kV)
            self._line_data.append(line_data)
        
        # Create all trafo models: 
        self._trafo_data = [] 
        for _, row in self._grid_trafos.iterrows(): 
            trafo_data = TrafoDataClass(S_base_mva=row["S_nom"], V_n_hv=row["V_hv_kV"], V_n_lv=row["V_lv_kV"], V_SCH=row["V_SCH_pu"], 
                                        P_Cu=row["P_Cu_pu"], I_E=row["I_E_pu"], P_Fe=row["P_Fe_pu"], idx_hv=row["idx_hv"], idx_lv=row["idx_lv"], 
                                        is_pu=True, tap_pos=row["tap_pos"], tap_change=row["tap_change"], tap_min=row["tap_min"], tap_max=row["tap_max"]
                                        )
            trafo_data = trafo_data.change_base(self.S_base_mva, self.V_base_kV)
            self._trafo_data.append(trafo_data)

        self._y_bus = self._create_y_bus()

    def _create_y_bus(self):
        y_bus = np.zeros((self.N_buses, self.N_buses), dtype=np.complex64)
        for line_data in self._line_data: 
            y_bus[line_data.idx_1, line_data.idx_2] -= line_data.y_series
            y_bus[line_data.idx_2, line_data.idx_1] -= line_data.y_series
            y_bus[line_data.idx_1, line_data.idx_1] += line_data.y_1_shunt
            y_bus[line_data.idx_2, line_data.idx_2] += line_data.y_2_shunt

        for trafo_data in self._trafo_data: 
            y_bus[trafo_data.idx_1, trafo_data.idx_2] -= trafo_data.y_series
            y_bus[trafo_data.idx_2, trafo_data.idx_1] -= trafo_data.y_series
            y_bus[trafo_data.idx_1, trafo_data.idx_1] += trafo_data.y_1_shunt
            y_bus[trafo_data.idx_2, trafo_data.idx_2] += trafo_data.y_2_shunt

        for i, y_row in enumerate(y_bus): 
            y_bus[i, i] += - sum(y_row[:i]) - sum(y_row[i+1:])
        return y_bus

    def get_Y_bus(self) -> Sequence[float]: 
        """Returns Y_bus"""
        return self._y_bus
    
    def get_PQ_mask(self) -> Tuple[Sequence[float], Sequence[float]]:
        """
        Returns (P_mask, Q_mask) \n 
        Used for obtaining the correct powers during calculation. size(P_mask) = (N_PQ+N_PV, ), size(Q_mask) = (N_PQ, )"""
        # P_mask = np.zeros(self.N_buses, dtype=int)
        # Q_mask = np.zeros(self.N_buses, dtype=int) 
        P_mask = [] 
        Q_mask = []
        gen_bus_idx = []

        for _, row in self._grid_gens.iterrows(): 
            if row["is_slack"] == 0:
                # P_mask[row["bus_idx"]] = 1
                P_mask.append(row["bus_idx"])
                gen_bus_idx.append(row["bus_idx"])

        for _, row in self._grid_loads.iterrows(): 
            P_mask.append(row["bus_idx"])
            if not row["bus_idx"] in gen_bus_idx:
                Q_mask.append(row["bus_idx"])
        
        return np.unique(P_mask).astype(int), np.unique(Q_mask).astype(int)

    def get_V_delta_mask(self) -> Tuple[Sequence[float], Sequence[float], float]: 
        """
        Returns (V_mask, delta_mask, N_delta) \n
        Used for determening the variables in which to solve for when doing power flow. \n
        Size of each mask = (N_buses, ) """
        # V_mask = np.zeros(self.N_buses, dtype=int)
        # delta_mask = np.zeros(self.N_buses, dtype=int) 
        V_mask = [] 
        delta_mask = []
        self.N_delta = 0
        self._gen_bus_idx = []

        for _, row in self._grid_gens.iterrows(): 
            if row["is_slack"] == 0:
                self.N_delta += 1 
                delta_mask.append(row["bus_idx"])
                self._gen_bus_idx.append(row["bus_idx"])

        for _, row in self._grid_loads.iterrows(): 
            if not row["bus_idx"] in self._gen_bus_idx:
                V_mask.append(row["bus_idx"])
                delta_mask.append(row["bus_idx"])
                self.N_delta += 1

        return np.unique(V_mask).astype(int), np.unique(delta_mask).astype(int), self.N_delta

    def get_V_delta_vals(self) -> Tuple[Sequence[float], Sequence[float]]:
        """
        Returns two vectors of size (N_buses) which contains the known values for V and delta. If unknown, initialize to 1.0 for voltage, and 0.0 for delta. """
        delta_vals = np.zeros(self.N_buses) 
        V_vals = np.ones(self.N_buses) 
        for _, row in self._grid_gens.iterrows(): 
            V_vals[row["bus_idx"]] = row["v_set_pu"]
        return V_vals, delta_vals

    def get_PQ_vals(self) -> Tuple[Sequence[float], Sequence[float]]: 
        """
        Returns two vectors of size (N_buses) which contains the known values for P and Q. If unknown, initialize to 0.0 for both. """
        P_vals = np.zeros(self.N_buses) 
        Q_vals = np.zeros(self.N_buses) 
        for _, row in self._grid_gens.iterrows(): 
            if row["is_slack"] == 0:
                P_vals[row["bus_idx"]] += row["p_set_mw"]/self.S_base_mva

        for _, row in self._grid_loads.iterrows(): 
            P_vals[row["bus_idx"]] -= row["p_nom_mw"]/self.S_base_mva
            Q_vals[row["bus_idx"]] -= row["q_nom_mvar"]/self.S_base_mva

        return P_vals, Q_vals 
    
    def change_P_gen(self, indices: Sequence[int], P_vals_mw: Sequence[float]): 
        """Note: The indices are the generator indices in order from the Excel sheet. """ 
        for P_new, idx in zip(P_vals_mw, indices): 
            self._grid_gens.at[idx, "p_set_mw"] = P_new

    def change_V_gen(self, indices: Sequence[int], V_vals: Sequence[float]):
        """Note: The indices are the generator indices in order from the Excel sheet. """ 
        for V_new, idx in zip(V_vals, indices): 
            self._grid_gens.at[idx, "v_set_pu"] = V_new           

    def change_P_load(self, indices: Sequence[int], P_vals_mw: Sequence[float]): 
        """Note: The indices are the load indices in order from the Excel sheet. """ 
        for P_new, idx in zip(P_vals_mw, indices): 
            # self._grid_loads.loc[idx, "p_nom_mw"] = P_new
            self._grid_loads.at[idx, "p_nom_mw"] = P_new

    def change_Q_load(self, indices: Sequence[int], Q_vals_mw: Sequence[float]): 
        """Note: The indices are the load indices in order from the Excel sheet. """ 
        for Q_new, idx in zip(Q_vals_mw, indices): 
            # self._grid_loads.loc[idx, "q_nom_mvar"] = Q_new
            self._grid_loads.at[idx, "q_nom_mvar"] = Q_new

class GridModel: 
    def __init__(self, grid_data: GridDataClass): 
        self.md = grid_data 
        self.y_bus = self.md.get_Y_bus() 
        self.P_mask, self.Q_mask = self.md.get_PQ_mask() 
        self.V_mask, self.delta_mask, _ = self.md.get_V_delta_mask() 

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
    
    def _calc_pf(self) -> PowerFlowResult: 
        X0 = np.zeros(len(self.delta_mask) + len(self.V_mask))
        X0[self.md.N_delta:] = 1.0 # flat voltage start
        pf_eqs = self._setup_pf() 
        sol = root(pf_eqs, X0, tol=1e-8)
        return sol 
    
    def _get_pf_sol(self, sol: OptimizeResult): 
        V_vals, delta_vals = self.md.get_V_delta_vals()
        delta_vals[self.delta_mask] = sol.x[:self.md.N_delta]
        V_vals[self.V_mask] = sol.x[self.md.N_delta:]
        P_calc, Q_calc = self._do_pf(V_vals, delta_vals, self.y_bus) 
        return PowerFlowResult(P_calc, Q_calc, V_vals, delta_vals, self.md.S_base_mva)
    
    def powerflow(self) -> PowerFlowResult: 
        """ 
        Does a power flow calculation based on the values specified in excel. 
        
        Returns 
        ---------
        PowerFlowResult 
        """
        sol = self._calc_pf()
        if not sol.success: 
            print(sol)
        sol = self._get_pf_sol(sol) 
        return sol 
                

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
