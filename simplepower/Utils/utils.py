import numpy as np
import pandas as pd
from ..Dataclasses.GridDataClass import GridDataClass
from copy import deepcopy

class PowerFlowResult: 
    def __init__(self, P_calc, Q_calc, V_buses, d_buses, S_base, scipy_sol): 
        self.P_calc = P_calc * S_base
        self.Q_calc = Q_calc * S_base
        self.V_buses = V_buses 
        self.d_buses = d_buses
        self.S_base = S_base
        self.scipy_sol = scipy_sol

    def __repr__(self): 
        str1 = f"P_calc = {np.round(self.P_calc, 4)} MW \n" 
        str2 = f"Q_calc = {np.round(self.Q_calc, 4)} Mvar \n" 
        str3 = f"V_buses = {np.round(self.V_buses, 6)} pu \n" 
        str4 = f"d_buses = {np.round(self.d_buses*180/np.pi, 6)} deg \n" 
        return str1 + str2 + str3 + str4
    
    def get_P_losses(self): 
        """Returns P_loss_MW"""
        return np.sum(self.P_calc)
    
    def get_sol_df(self): 
        sol = {"P_inj MW": self.P_calc, "Q_inj_Mvar": self.Q_calc, "V_bus_pu": self.V_buses, "delta_bus_deg": self.d_buses*180/np.pi}
        return pd.DataFrame(sol)
    
    def store_json(self, filename: str): #
        """Stores the power flow results into a json file at specified location. """
        self.get_sol_df().to_json(filename) 

def convert_PV_to_PQ_grid(grid_data: GridDataClass, pf_res: PowerFlowResult): 
    grid_data_PQ = deepcopy(grid_data) 
    # Adding a load for each generator after the power flow*
    N_loads = len(grid_data_PQ._grid_loads)
    idx = 0
    for _, gen in grid_data_PQ._grid_gens.iterrows(): 
        if gen["is_slack"] != 1:
            new_data = {"name": gen["name"], "v_nom_kv": grid_data.V_base_kV, 
                        "s_base_mva": gen["S_rated_mva"], "v_nom_pu": 1.0, 
                        "p_nom_mw": -pf_res.P_calc[gen["bus_idx"]], 
                        "q_nom_mvar": -pf_res.Q_calc[gen["bus_idx"]], 
                        "bus_idx": gen["bus_idx"], "g_shunt_pu": 0.0, 
                        "b_shunt_pu": 0.0}
            grid_data_PQ._grid_loads.loc[N_loads+idx] = pd.Series(new_data) 
            idx += 1 
            
    # Code for removing all generators except the slack 
    idx_slack = np.argmax(grid_data_PQ._grid_gens["is_slack"] == 1)
    N_gens = len(grid_data_PQ._grid_gens)
    gen_idx = [i for i in range(N_gens) if i != idx_slack]
    grid_data_PQ._grid_gens.drop(index=gen_idx, inplace=True)
    return grid_data_PQ