from typing import Optional, Sequence, Tuple
import pandas as pd 
import numpy as np 
from branch_model import LineDataClass, TrafoDataClass
from enum import Enum

from import_grid_model import get_item_dict

"""
Two methods of import is supported at the moment: 
1) Manual excel sheet insertion + import excel file through pandas
2) IEEE Common Data Format and .txt file. 

Both methods should end up with five pandas dataframes: _grid_buses, _grid_lines, _grid_trafos, _grid_loads, _grid_gens

_grid_buses: [bus_idx	name	v_nom_kv]
_grid_lines: [name	v_nom_kv	length_km	r_ohm_per_km	x_ohm_per_km	c_uf_per_km	from_bus_idx	to_bus_idx	is_pu]
_grid_trafos: [name	S_nom	V_hv_kV	V_lv_kV	V_SCH_pu	P_Cu_pu	I_E_pu	P_Fe_pu	idx_hv	idx_lv	tap_pos	tap_change	tap_min	tap_max]
_grid_loads: [name	v_base_kV	s_base_mva	v_nom_pu	p_nom_mw	q_nom_mvar	bus_idx	g_shunt_pu	b_shunt_pu]
_grid_gens: [name	S_rated_mva	v_set_pu	p_set_mw	bus_idx	is_slack]
"""

class FileType(Enum): 
    Excel = 0, 
    IEEE = 1, 

class ExcelImport: 
    def __init__(self, filename): 
        self.filename = filename
        self._real_excel(filename) 
    
    def _real_excel(self, filename): 
        self._grid_buses = pd.read_excel(filename, sheet_name="busbars")
        self._grid_lines = pd.read_excel(filename, sheet_name="lines")
        self._grid_trafos  = pd.read_excel(filename, sheet_name="trafos")
        self._grid_loads = pd.read_excel(filename, sheet_name="loads")
        self._grid_gens  = pd.read_excel(filename, sheet_name="gens")

    def get_data(self): 
        return (self._grid_buses, 
                self._grid_lines, 
                self._grid_trafos,
                self._grid_loads,
                self._grid_gens)
    

class IEEEImport: 
    def __init__(self, filename): 
        self.filename = filename 
        self._read_IEEE(filename)

    def _read_IEEE(self, filename): 
        self.bus_data, self.branch_data, self.S_base = get_item_dict(self.filename)

    def _get_bus_data(self): 
        bus_data = {}
        bus_data["bus_idx"] = self.bus_data["bus_num"] - 1
        bus_data["name"] = self.bus_data["name"]
        bus_data["v_nom_kv"] = self.bus_data["base_kV"]
        return bus_data 

    def _get_line_data(self): 
        line_data = {}
        line_idx = np.where((self.branch_data["tap_ratio_final"] == 0.0))[0]
        line_data["name"] = np.array([f"Line {i}" for i in range(len(line_idx))])
        line_data["length_km"] = 1.0 
        line_data["r_ohm_per_km"] = self.branch_data["r_pu"][line_idx]
        line_data["x_ohm_per_km"] = self.branch_data["x_pu"][line_idx]
        line_data["c_uf_per_km"] = 2*np.pi*self.branch_data["b_pu"][line_idx]
        line_data["from_bus_idx"] = self.branch_data["tap_bus"][line_idx]-1
        line_data["to_bus_idx"] = self.branch_data["Z_bus"][line_idx]-1
        line_data["is_pu"] = [1 for _ in range(len(line_idx))] 
        line_data["v_nom_kv"] = self.bus_data["base_kV"][line_data["from_bus_idx"]]
        return line_data 
    
    def _get_trafo_data(self): 
        trafo_data = {}
        trafo_idx = np.where((self.branch_data["tap_ratio_final"] > 0.0))[0]
        trafo_data["name"] = np.array([f"Trafo {i}" for i in range(len(trafo_idx))])
        trafo_data["S_nom"] = np.array([self.S_base for _ in range(len(trafo_idx))])
        trafo_r = self.branch_data["r_pu"][trafo_idx]
        trafo_x = self.branch_data["x_pu"][trafo_idx]
        trafo_data["V_SCH_pu"] = np.sqrt(trafo_r**2 + trafo_x**2)
        trafo_data["P_Cu_pu"] = trafo_r
        trafo_data["I_E_pu"] = np.zeros(len(trafo_idx))
        trafo_data["P_Fe_pu"] = np.zeros(len(trafo_idx))
        trafo_data["idx_hv"] = self.branch_data["tap_bus"][trafo_idx]-1
        trafo_data["idx_lv"] = self.branch_data["Z_bus"][trafo_idx]-1
        trafo_data["tap_pos"] = np.ones(len(trafo_idx))
        trafo_data["tap_change"] = 1 - self.branch_data["tap_ratio_final"][trafo_idx]
        trafo_data["tap_min"] = self.branch_data["tap_min"][trafo_idx]
        trafo_data["tap_max"] = self.branch_data["tap_max"][trafo_idx]
        trafo_data["V_hv_kV"] = self.bus_data["base_kV"][trafo_data["idx_hv"]]
        trafo_data["V_lv_kV"] = self.bus_data["base_kV"][trafo_data["idx_lv"]]
        return trafo_data
    
    def _get_load_data(self): 
        load_data = {}
        load_idx = np.where(np.logical_or(self.bus_data["load_mw"] > 0.0, self.bus_data["load_mvar"] > 0.0))[0]
        load_data["name"] = np.array([f"Load bus {idx}" for idx in load_idx])
        load_data["v_base_kV"] = self.bus_data["base_kV"][load_idx]
        load_data["s_base_mva"] = np.ones(len(load_idx))*self.S_base
        load_data["v_nom_pu"] = np.ones(len(load_idx))
        load_data["p_nom_mw"] = self.bus_data["load_mw"][load_idx]
        load_data["q_nom_mvar"] = self.bus_data["load_mvar"][load_idx]
        load_data["bus_idx"] = load_idx
        load_data["g_shunt_pu"] = self.bus_data["shunt_g_pu"][load_idx]
        load_data["b_shunt_pu"] = self.bus_data["shunt_b_pu"][load_idx]
        return load_data
    
    def _get_gen_data(self): 
        gen_data = {}
        gen_idx = np.where(self.bus_data["gen_mw"] > 0.0)[0]
        gen_data["name"] = np.array([f"Gen {idx}" for idx in gen_idx])
        gen_data["S_rated_mva"] = np.ones(len(gen_idx))*self.S_base
        gen_data["v_set_pu"] = self.bus_data["final_v"][gen_idx]
        gen_data["p_set_mw"] = self.bus_data["gen_mw"][gen_idx]
        gen_data["bus_idx"] = gen_idx 
        gen_type = self.bus_data["type"][gen_idx]
        gen_data["is_slack"] = np.array(np.equal(gen_type, 3), dtype=int)
        return gen_data 

    def get_data(self): 
        bus_data = self._get_bus_data()
        line_data = self._get_line_data()
        trafo_data = self._get_trafo_data() 
        load_data = self._get_load_data() 
        gen_data = self._get_gen_data() 

        bus_data = pd.DataFrame(bus_data)
        line_data = pd.DataFrame(line_data)
        trafo_data = pd.DataFrame(trafo_data)
        load_data = pd.DataFrame(load_data)
        gen_data = pd.DataFrame(gen_data)

        return bus_data, line_data, trafo_data, load_data, gen_data 


class GridDataClass: 
    def __init__(self, filename: str, filetype: FileType, f_nom: float, V_init: Optional[Sequence[float]]=None, delta_init: Optional[Sequence[float]]=None):         
        self._read_data(filename, filetype)
        self._set_base_vals(f_nom)
        self._set_init_condition(V_init, delta_init)
        self._set_line_data()
        self._set_trafo_data()
        self._set_shunt_data()
        self._y_bus = self._create_y_bus()

    def _read_data(self, filename, filetype): 
        match filetype: 
            case FileType.Excel: 
                data = ExcelImport(filename)
            case FileType.IEEE: 
                data = IEEEImport(filename)

        (self._grid_buses, self._grid_lines, self._grid_trafos, self._grid_loads, self._grid_gens) = data.get_data()

    def _set_base_vals(self, f_nom): 
        self.S_base_mva = self._grid_gens["S_rated_mva"].sum()
        self.V_base_kV = self._grid_buses["v_nom_kv"].max()
        self.f_nom = f_nom
        self.N_buses = self._grid_buses.shape[0] 

    def _set_init_condition(self, V_init, delta_init): 
        if V_init is None: 
            self.V_init = np.ones(self.N_buses)
        else: 
            self.V_init = V_init
        if delta_init is None: 
            self.delta_init = np.zeros(self.N_buses)
        else: 
            self.delta_init = delta_init
        
    def _set_line_data(self): 
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

    def _set_trafo_data(self): 
        self._trafo_data = [] 
        for _, row in self._grid_trafos.iterrows(): 
            trafo_data = TrafoDataClass(S_base_mva=row["S_nom"], V_n_hv=row["V_hv_kV"], V_n_lv=row["V_lv_kV"], V_SCH=row["V_SCH_pu"], 
                                        P_Cu=row["P_Cu_pu"], I_E=row["I_E_pu"], P_Fe=row["P_Fe_pu"], idx_hv=row["idx_hv"], idx_lv=row["idx_lv"], 
                                        is_pu=True, tap_pos=row["tap_pos"], tap_change=row["tap_change"], tap_min=row["tap_min"], tap_max=row["tap_max"]
                                        )
            trafo_data = trafo_data.change_base(self.S_base_mva, self.V_base_kV)
            self._trafo_data.append(trafo_data)

    def _set_shunt_data(self): 
        self._shunt_data = []
        for _, row in self._grid_loads.iterrows(): 
            factor = (row["v_base_kV"]**2/row["s_base_mva"]) * (self.S_base_mva/(self.V_base_kV**2))
            g_shunt = row["g_shunt_pu"]
            b_shunt = row["b_shunt_pu"]

            g = g_shunt / factor # Change the base value 
            b = b_shunt / factor 
            self._shunt_data.append(g - 1j*b)

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

        for i, row in self._grid_loads.iterrows(): 
            idx = row["bus_idx"]
            y_bus[idx, idx] -= self._shunt_data[i]

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
        delta_vals = self.delta_init.copy()
        V_vals = self.V_init.copy()
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