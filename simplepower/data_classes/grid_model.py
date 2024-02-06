from typing import Optional, Sequence, Tuple
import pandas as pd 
import numpy as np 
from enum import Enum
from copy import deepcopy 

from .branch_model import LineDataClass, TrafoDataClass
from ..component_models import BasePQGenerator, BasePVGenerator, BasePQLoad, BaseComponentModel
from ..utils import BaseTimeSeries, PQVD

class ExcelImport: 
    def __init__(self, filename): 
        """
        Import network data through reading an excel file. 

        The import should create six pandas dataframes in the grid dataclass: _grid_buses, _grid_lines, _grid_trafos, _grid_loads, _grid_gens, _grid_static_gens

        _grid_buses: [bus_idx	name	v_nom_kv]
        _grid_lines: [name	v_nom_kv	length_km	r_ohm_per_km	x_ohm_per_km	c_uf_per_km	from_bus_idx	to_bus_idx	is_pu]
        _grid_trafos: [name	S_nom	V_hv_kV	V_lv_kV	V_SCH_pu	P_Cu_pu	I_E_pu	P_Fe_pu	idx_hv	idx_lv	tap_pos	tap_change	tap_min	tap_max]
        _grid_loads: [name	v_nom_kv	s_base_mva	v_nom_pu	p_nom_mw	q_nom_mvar	bus_idx	g_shunt_pu	b_shunt_pu]
        _grid_gens: [name	S_rated_mva	v_set_pu	p_set_mw	bus_idx	is_slack]
        """
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

    
class GridDataClass: 
    def __init__(self, filename: str, f_nom: float, models: list[BaseComponentModel]=[],
                 V_init: Optional[Sequence[float]]=None, delta_init: Optional[Sequence[float]]=None, 
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
        self._define_PQV_models(models)

    def _re_init(self, f_nom: float, V_init: Optional[Sequence[float]]=None, delta_init: Optional[Sequence[float]]=None, 
                 S_base_mva: Optional[float]=None, V_base_kV: Optional[float]=None): 
        self._set_base_vals(f_nom, S_base_mva, V_base_kV)
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
            self.S_base_mva = self._grid_gens["S_rated_mva"].sum() + self._grid_static_gens["S_rated_mva"].sum()
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
            trafo_data = TrafoDataClass(S_base_mva=row["S_nom"], V_n_hv=row["V_hv_kV"], V_n_lv=row["V_lv_kV"], V_base_kV=row["v_base_kV"], V_SCH=row["V_SCH_pu"],
                                        P_Cu=row["P_Cu_pu"], I_E=row["I_E_pu"], P_Fe=row["P_Fe_pu"], idx_hv=row["idx_hv"], idx_lv=row["idx_lv"], 
                                        is_pu=True, tap_pos=row["tap_pos"], tap_change=row["tap_change"], tap_min=row["tap_min"], tap_max=row["tap_max"], 
                                        z_leak_hv=z_hv)
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
    
    def _define_PQV_models(self, other_models: list[BaseComponentModel]): 
        models = []
        for idx, gen in self._grid_gens.iterrows(): 
            gen_profile = BaseTimeSeries(gen["p_set_mw"], gen["v_set_pu"])
            model = BasePVGenerator(gen["bus_idx"], gen_profile)
            models.append(model) 

        for idx, gen in self._grid_static_gens.iterrows(): 
            gen_profile = BaseTimeSeries(gen["p_set_mw"], gen["q_set_mvar"])
            model = BasePQGenerator(gen["bus_idx"], gen_profile)
            models.append(model) 

        for idx, load in self._grid_loads.iterrows(): 
            load_profile = BaseTimeSeries(load["p_nom_mw"], load["q_nom_mvar"])
            model = BasePQLoad(load["bus_idx"], load_profile)
            models.append(model) 

        self.base_models = self._get_model_dict(models)
        self.added_models = self._get_model_dict(other_models)
    
    def _give_added_models(self, added_models: dict): 
        self.added_models = deepcopy(added_models)

    def _get_model_dict(self, models: list[BaseComponentModel]):
        levels = []
        for model in models: 
            levels.append(model.level)
        levels = np.unique(levels) 
        model_dict = {level: [] for level in levels} 
        for model in models: 
            model_dict[model.level].append(model)
        return model_dict

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

    def _get_V_delta_vals(self, time_index: Optional[int]=0) -> Tuple[Sequence[float], Sequence[float]]:
        V_vals = self.V_init.copy()
        V_vals = np.zeros(self.N_buses, dtype=float)
        delta_vals = self.delta_init.copy()
        for level, models in self.base_models.items():
            for model in models:
                V_vals[model.bus_idx] += model.V_add_equation(None, time_index) # TODO: Dont need pqvd in the v_add equations
        for level, models in self.added_models.items(): 
            for model in models: 
                V_vals[model.bus_idx] += model.V_add_equation(None, time_index)
        return V_vals, delta_vals

    def _get_PQ_vals(self, pqvd: PQVD, time_index: Optional[int]=0) -> Tuple[Sequence[float], Sequence[float]]: 
        P_vals = np.zeros(self.N_buses) 
        Q_vals = np.zeros(self.N_buses) 
        for level, models in self.base_models.items():
            for model in models: 
                P_vals[model.bus_idx] += model.P_add_equation(pqvd, time_index)/self.S_base_mva
                Q_vals[model.bus_idx] += model.Q_add_equation(pqvd, time_index)/self.S_base_mva
                
        for level, models in self.added_models.items():
            for model in models: 
                P_vals[model.bus_idx] += model.P_add_equation(pqvd, time_index)/self.S_base_mva
                Q_vals[model.bus_idx] += model.Q_add_equation(pqvd, time_index)/self.S_base_mva
        return P_vals, Q_vals 
    
    def change_model_data(self, bus_idx: int, model_type: BaseComponentModel, 
                          new_data: BaseTimeSeries): 
        for level, models in self.base_models.items(): 
            for model in models: 
                if model.bus_idx == bus_idx: 
                    if isinstance(model, model_type): 
                        model.replace_data(new_data)
        for level, models in self.added_models.items(): 
            for model in models: 
                if model.bus_idx == bus_idx: 
                    if isinstance(model, model_type): 
                        model.replace_data(new_data)

    def replace_model(self, old_model_type: BaseComponentModel, 
                      new_model: BaseComponentModel): 
        for level, models in self.base_models.items(): 
            for i, model in enumerate(models): 
                if model.bus_idx == new_model.bus_idx: 
                    if isinstance(model, old_model_type): 
                        models.remove(models[i])
                        self.add_model(new_model)
        for level, models in self.added_models.items(): 
            for i, model in enumerate(models): 
                if model.bus_idx == new_model.bus_idx: 
                    if isinstance(model, old_model_type): 
                        models[i] = new_model

    def add_model(self, new_model: BaseComponentModel):
        if new_model.level in self.added_models: 
            self.added_models[new_model.level].append(new_model) 
        else: 
            self.added_models[new_model.level] = [new_model]

    def remove_model(self, bus_idx: int, model_type: BaseComponentModel): 
        for level, models in self.base_models.items(): 
            for i, model in enumerate(models): 
                if model.bus_idx == bus_idx: 
                    if isinstance(model, model_type): 
                        models.remove(models[i])
        for level, models in self.added_models.items(): 
            for i, model in enumerate(models): 
                if model.bus_idx == bus_idx: 
                    if isinstance(model, model_type): 
                        models.remove(models[i])

    def change_lines(self, line_idx: Sequence[int], 
                    r_new: Optional[Sequence[float]]=None,
                    x_new: Optional[Sequence[float]]=None,
                    length_new: Optional[Sequence[float]]=None, 
                    c_new: Optional[Sequence[float]]=None, 
                    is_pu_new: Optional[Sequence[bool]]=None): 
        if not r_new is None: 
            for idx, r in zip(line_idx, r_new):
                self._grid_lines.at[idx, "r_ohm_per_km"] = r 
        if not x_new is None: 
            for idx, x in zip(line_idx, x_new):
                self._grid_lines.at[idx, "x_ohm_per_km"] = x 
        if not length_new is None: 
            for idx, length in zip(line_idx, length_new):
                self._grid_lines.at[idx, "length_km"] = length 
        if not c_new is None: 
            for idx, c in zip(line_idx, c_new):
                self._grid_lines.at[idx, "c_uf_per_km"] = c 
        if not is_pu_new is None: 
            for idx, is_pu in zip(line_idx, is_pu_new):
                self._grid_lines.at[idx, "is_pu"] = is_pu 
        
        self._re_init(f_nom=self.f_nom, S_base_mva=self.S_base_mva)
