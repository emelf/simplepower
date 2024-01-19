import numpy as np
import pandas as pd
from ..Dataclasses import GridDataClass
from ..PowerFlowModels import GridModel
from copy import deepcopy

def convert_PV_to_PQ_grid(grid_data: GridDataClass): 
    grid_data_PQ = deepcopy(grid_data) 
    grid_model = GridModel(grid_data_PQ)
    pf_res = grid_model.powerflow()
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
    return grid_data_PQ