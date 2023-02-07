import numpy as np 
import pandas as pd 
from dataclasses import dataclass 
from scipy.optimize import root 
from typing import Tuple, Sequence, Callable
import cmath as cm


# @dataclass
class GridDataClass: 
    def __init__(self, filename: str, f_nom: float): 
        self._grid_buses = pd.read_excel(filename, sheet_name="busbars")
        self._grid_lines = pd.read_excel(filename, sheet_name="lines")
        self._grid_loads = pd.read_excel(filename, sheet_name="loads")
        self._grid_gens  = pd.read_excel(filename, sheet_name="gens")
        self.S_base_mva = self._grid_gens["S_rated_mva"].sum()
        self.f_nom = f_nom
        self.N_buses = self._grid_buses.shape[0]
    
    # def __post_init__(self): 
        self._y_bus = self._create_y_bus() 

    def _create_y_bus(self):
        y_bus = np.zeros((self.N_buses, self.N_buses), dtype=np.complex64)
        for _, row in self._grid_lines.iterrows(): 
            v_base = row["v_nom_kv"]
            r = row["r_ohm_per_km"]
            x = row["x_ohm_per_km"] 
            c = row["c_uf_per_km"]
            i = row["from_bus_idx"]
            j = row["to_bus_idx"]
            z_base = v_base**2/self.S_base_mva
            z_line = (r + 1j*x)/z_base
            b_line_half = 2*np.pi*self.f_nom*c*1e-6 / 2 * z_base #TODO: Find the frequency from somewhere else

            y_bus[i, j] = -(z_line**-1)
            y_bus[j, i] = -(z_line**-1)
            y_bus[i, i] += b_line_half
            y_bus[j, j] += b_line_half

        for i, y_row in enumerate(y_bus): 
            y_bus[i, i] = - sum(y_row) + y_row[i] 
        return y_bus

    def get_Y_bus(self) -> Sequence[float]: 
        """Returns Y_bus"""
        return self._y_bus
    
    def get_PQ_mask(self) -> Tuple[Sequence[float], Sequence[float]]:
        """
        Returns (P_mask, Q_mask) \n 
        Used for obtaining the correct powers during calculation. size(P_mask) = (N_buses, ), size(Q_mask) = (N_buses, )"""
        # P_mask = np.zeros(self.N_buses, dtype=int)
        # Q_mask = np.zeros(self.N_buses, dtype=int) 
        P_mask = [] 
        Q_mask = []

        for _, row in self._grid_gens.iterrows(): 
            if row["is_slack"] == 0:
                # P_mask[row["bus_idx"]] = 1
                P_mask.append(row["bus_idx"])

        for _, row in self._grid_loads.iterrows(): 
            # P_mask[row["bus_idx"]] = 1
            # Q_mask[row["bus_idx"]] = 1
            P_mask.append(row["bus_idx"])
            Q_mask.append(row["bus_idx"])
        
        return np.array(P_mask, dtype=int), np.array(Q_mask, dtype=int) 

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

        for _, row in self._grid_gens.iterrows(): 
            if row["is_slack"] == 0:
                # delta_mask[row["bus_idx"]] = 1
                self.N_delta += 1
                delta_mask.append(row["bus_idx"])

        for _, row in self._grid_loads.iterrows(): 
            # V_mask[row["bus_idx"]] = 1
            # delta_mask[row["bus_idx"]] = 1
            V_mask.append(row["bus_idx"])
            delta_mask.append(row["bus_idx"])
            self.N_delta += 1
        V_mask = np.array(V_mask, dtype=int)
        delta_mask = np.array(V_mask, dtype=int)
        return delta_mask, V_mask, self.N_delta

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
                P_vals[row["bus_idx"]] = row["p_set_mw"]/self.S_base_mva

        for _, row in self._grid_loads.iterrows(): 
            P_vals[row["bus_idx"]] = row["p_nom_mw"]/self.S_base_mva
            Q_vals[row["bus_idx"]] = row["q_nom_mvar"]/self.S_base_mva

        return -P_vals, -Q_vals 

class GridModel: 
    def __init__(self, grid_data: GridDataClass): 
        self.md = grid_data 
        self.y_bus = self.md.get_Y_bus() 
        self.P_mask, self.Q_mask = self.md.get_PQ_mask() 
        self.V_mask, self.delta_mask, _ = self.md.get_V_delta_mask() 

    def do_pf(self, V_vals, delta_vals, y_bus): 
        # Convert to a vector of complex voltages 
        V_vec = V_vals*np.cos(delta_vals) + V_vals*np.sin(delta_vals)*1j
        # V_mat = np.eye(len(V_vec))*V_vec  
        S_conj = V_vec.conj() * (y_bus @ V_vec)
        P_calc = S_conj.real 
        Q_calc = -S_conj.imag 
        return P_calc, Q_calc

    def setup_pf(self) -> Callable[[np.ndarray], np.ndarray]:
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

            P_calc, Q_calc = self.do_pf(_V_vals, _delta_vals, self.y_bus)

            P_root = (P_calc - P_vals)[self.P_mask]
            Q_root = (Q_calc - Q_vals)[self.Q_mask]

            S_root = np.zeros(len(P_root)+len(Q_root)) # Do this initialization to be numba compatible 
            S_root[:len(P_root)] = P_root 
            S_root[len(P_root):] = Q_root
            return S_root 
        
        return pf_eqs 
    
    # def get_init_

            
