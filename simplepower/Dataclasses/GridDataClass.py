from typing import Optional, Sequence, Tuple
import pandas as pd 
import numpy as np 
from enum import Enum

from .BranchModels import LineDataClass, TrafoDataClass

"""
Two methods of import is supported at the moment: 
1) Manual excel sheet insertion + import excel file through pandas
2) IEEE Common Data Format and .txt file. 

Both methods should end up with five pandas dataframes: _grid_buses, _grid_lines, _grid_trafos, _grid_loads, _grid_gens

_grid_buses: [bus_idx	name	v_nom_kv]
_grid_lines: [name	v_nom_kv	length_km	r_ohm_per_km	x_ohm_per_km	c_uf_per_km	from_bus_idx	to_bus_idx	is_pu]
_grid_trafos: [name	S_nom	V_hv_kV	V_lv_kV	V_SCH_pu	P_Cu_pu	I_E_pu	P_Fe_pu	idx_hv	idx_lv	tap_pos	tap_change	tap_min	tap_max]
_grid_loads: [name	v_nom_kv	s_base_mva	v_nom_pu	p_nom_mw	q_nom_mvar	bus_idx	g_shunt_pu	b_shunt_pu]
_grid_gens: [name	S_rated_mva	v_set_pu	p_set_mw	bus_idx	is_slack]
"""

class FileType(Enum): # NOTE: Being deprecated
    Excel = 0, 
    IEEE = 1, 
    JSON = 2


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
        self._grid_static_gens = pd.read_excel(filename, sheet_name="static gens")

    def get_data(self): 
        return (self._grid_buses, 
                self._grid_lines, 
                self._grid_trafos,
                self._grid_loads,
                self._grid_gens, 
                self._grid_static_gens)
    

class JSONImport: # TODO
    pass 
    
class GridDataClass: 
    def __init__(self, filename: str, f_nom: float, V_init: Optional[Sequence[float]]=None, delta_init: Optional[Sequence[float]]=None, 
                 S_base_mva: Optional[float]=None, V_base_kV: Optional[float]=None):       
        self._read_data(filename)
        self._set_base_vals(f_nom, S_base_mva, V_base_kV)
        self._set_lim_vals()
        self._set_init_condition(V_init, delta_init)
        self._set_line_data()
        self._set_trafo_data()
        self._set_shunt_data()
        self._y_bus = self._create_y_bus()
        self._y_lines = self._create_y_lines()

    def _re_init(self, f_nom: float, V_init: Optional[Sequence[float]]=None, delta_init: Optional[Sequence[float]]=None, S_base_mva: Optional[float]=None): 
        self._set_base_vals(f_nom, S_base_mva)
        self._set_lim_vals()
        self._set_init_condition(V_init, delta_init)
        self._set_line_data()
        self._set_trafo_data()
        self._set_shunt_data()
        self._y_bus = self._create_y_bus()
        self._y_lines = self._create_y_lines()

    def _read_data(self, filename): 
        data = ExcelImport(filename)
        (self._grid_buses, self._grid_lines, self._grid_trafos, self._grid_loads, self._grid_gens, self._grid_static_gens) = data.get_data()

    def _set_base_vals(self, f_nom, S_base_mva, V_base_kV):
        if S_base_mva is None: 
            self.S_base_mva = self._grid_gens["S_rated_mva"].sum()
        else: self.S_base_mva = S_base_mva 
        if V_base_kV is None:
            self.V_base_kV = self._grid_buses["v_nom_kv"].max()
        else: 
            self.V_base_kV = V_base_kV
        self.f_nom = f_nom
        self.N_buses = self._grid_buses.shape[0] 
        self.I_bases_kA = self.S_base_mva / (self._grid_buses["v_nom_kv"].values * np.sqrt(3))

    def _set_lim_vals(self): 
        self.I_lims_pu = np.zeros((self.N_buses, self.N_buses))
        for idx, line in self._grid_lines.iterrows(): 
            I_lim = line["I_lim_A"]
            i = line["from_bus_idx"]
            j = line["to_bus_idx"]
            I_base = self.S_base_mva*1000/(np.sqrt(3)*self.V_base_kV)
            self.I_lims_pu[i, j] = self.I_lims_pu[j, i] = I_lim/I_base 

        for idx, trafo in self._grid_trafos.iterrows(): 
            s_nom = trafo["S_nom"]
            i = trafo["idx_hv"]
            j = trafo["idx_lv"]
            self.I_lims_pu[i, j] = self.I_lims_pu[j, i] = s_nom/self.S_base_mva

        self.columns = [f"From {i}" for i in range(self.N_buses)]
        self.indices = [f"To {i}" for i in range(self.N_buses)]
        self.I_lims_pu = pd.DataFrame(np.abs(self.I_lims_pu), columns=self.columns, index=self.indices)

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
            z_hv = 0.5 
            z_lv = 0.5
            trafo_data = TrafoDataClass(S_base_mva=row["S_nom"], V_n_hv=row["V_hv_kV"], V_n_lv=row["V_lv_kV"], V_base_kV=row["v_base_kV"], V_SCH=row["V_SCH_pu"],
                                        P_Cu=row["P_Cu_pu"], I_E=row["I_E_pu"], P_Fe=row["P_Fe_pu"], idx_hv=row["idx_hv"], idx_lv=row["idx_lv"], 
                                        is_pu=True, tap_pos=row["tap_pos"], tap_change=row["tap_change"], tap_min=row["tap_min"], tap_max=row["tap_max"], 
                                        z_leak_hv=z_hv, z_leak_lv=z_lv
                                        )
            trafo_data = trafo_data.change_base(self.S_base_mva, self.V_base_kV)
            self._trafo_data.append(trafo_data)

    def _set_shunt_data(self): 
        self._shunt_data = []
        for _, row in self._grid_loads.iterrows(): 
            factor = (self.S_base_mva / row["s_base_mva"]) #* (row["v_nom_kv"] / self.V_base_kV)**2
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
    
    def _create_y_lines(self): 
        y_lines = np.zeros((self.N_buses, self.N_buses), dtype=np.complex64)
        for line_data in self._line_data: 
            y_lines[line_data.idx_1, line_data.idx_2] += line_data.y_series
            y_lines[line_data.idx_2, line_data.idx_1] += line_data.y_series
            y_lines[line_data.idx_1, line_data.idx_1] += line_data.y_1_shunt
            y_lines[line_data.idx_2, line_data.idx_2] += line_data.y_2_shunt

        for trafo_data in self._trafo_data: 
            y_lines[trafo_data.idx_1, trafo_data.idx_2] += trafo_data.y_series
            y_lines[trafo_data.idx_2, trafo_data.idx_1] += trafo_data.y_series
            y_lines[trafo_data.idx_1, trafo_data.idx_1] += trafo_data.y_1_shunt
            y_lines[trafo_data.idx_2, trafo_data.idx_2] += trafo_data.y_2_shunt

        # for i, row in self._grid_loads.iterrows(): 
        #     idx = row["bus_idx"]
        #     y_lines[idx, idx] += self._shunt_data[i]

        return y_lines

    def get_Y_bus(self) -> Sequence[float]: 
        """Returns Y_bus"""
        return self._y_bus
    
    def get_Y_lines(self) -> Sequence[float]: 
        return self._y_lines
    
    def get_PQVd_mask(self) -> Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
        """
        Returns (P_mask, Q_mask, V_mask, d_mask) \n 
        Used for obtaining the correct powers during calculation. size(P_mask) = (N_PQ+N_PV-1), size(Q_mask) = (N_PQ, ), assuming one slack bus.
        P_mask, Q_mask -> Bus idx where P / Q is known \n 
        V_mask, d_mask -> Bus idx where V / delta is unknown. """
        P_mask = np.arange(0, self.N_buses, 1)
        Q_mask = np.arange(0, self.N_buses, 1)
        V_mask = np.arange(0, self.N_buses, 1)
        delta_mask = np.arange(0, self.N_buses, 1)

        gen_bus_idx = []
        gen_bus_idx_sl = []
        for _, row in self._grid_gens.iterrows(): 
            gen_bus_idx_sl.append(row["bus_idx"])
            if row["is_slack"] == 0:
                gen_bus_idx.append(row["bus_idx"])
            else: # In case slack bus
                P_mask = np.delete(P_mask, row["bus_idx"])
                delta_mask = np.delete(delta_mask, row["bus_idx"])

        Q_mask = np.delete(Q_mask, gen_bus_idx_sl)
        V_mask = np.delete(V_mask, gen_bus_idx_sl)
        self.N_delta = len(delta_mask)
        return P_mask, Q_mask, V_mask, delta_mask

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

        for _, row in self._grid_static_gens.iterrows(): 
            P_vals[row["bus_idx"]] += row["p_set_mw"]/self.S_base_mva
            Q_vals[row["bus_idx"]] += row["q_set_mvar"]/self.S_base_mva

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

    def change_P_static_gen(self, indices: Sequence[int], P_vals_mw: Sequence[float]): 
        """Note: The indices are the load indices in order from the Excel sheet. """ 
        for P_new, idx in zip(P_vals_mw, indices): 
            # self._grid_loads.loc[idx, "p_nom_mw"] = P_new
            self._grid_static_gens.at[idx, "p_set_mw"] = P_new

    def change_Q_static_gen(self, indices: Sequence[int], Q_vals_mw: Sequence[float]): 
        """Note: The indices are the load indices in order from the Excel sheet. """ 
        for Q_new, idx in zip(Q_vals_mw, indices): 
            # self._grid_loads.loc[idx, "q_nom_mvar"] = Q_new
            self._grid_static_gens.at[idx, "q_set_mvar"] = Q_new