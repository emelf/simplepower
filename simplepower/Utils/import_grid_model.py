import numpy as np

bus_locs = {"bus_num": (1, 5), "name": (5, 19), "area": (19, 21), "loss_zone": (21, 24), "type": (25, 27), 
            "final_v": (27, 33), "final_delta": (34, 41), "load_mw": (41, 50), "load_mvar": (50, 60), "gen_mw": (60, 68), "gen_mvar": (68, 75), 
            "base_kV": (77, 84), "v_desire": (84, 91), "q_max": (91, 99), "q_min": (99, 106), "shunt_g_pu": (107, 115), "shunt_b_pu": (115, 123), 
            "remote_bus": (124, 128)}
bus_types = {"bus_num": int, "name": str, "area": int, "loss_zone": int, "type": int, 
            "final_v": float, "final_delta": float, "load_mw": float, "load_mvar": float, "gen_mw": float, "gen_mvar": float, 
            "base_kV": float, "v_desire": float, "q_max": float, "q_min": float, "shunt_g_pu": float, "shunt_b_pu": float, 
            "remote_bus": int}

branch_locs = {"tap_bus": (1, 5), "Z_bus": (6,9), "area": (10, 13), "loss_zone": (13, 15), "circuit": (16, 18), "type": (18, 21), 
               "r_pu": (20, 29), "x_pu": (30, 40), "b_pu": (41, 50), "line_MVA_1": (51, 55), "line_MVA_2": (57, 61), "line_MVA_3": (63, 67), 
               "control_bus": (69, 72), "side": (73, 75), "tap_ratio_final": (77, 82), "phase_shift": (84, 90), "tap_min": (90, 97), "tap_max": (98, 104), 
               "step_size": (106, 111), "min_V/MVA": (112, 117), "max_V/MVA": (119, 125)}
branch_types = {"tap_bus": int, "Z_bus": int, "area": int, "loss_zone": int, "circuit": int, "type": int, 
               "r_pu": float, "x_pu": float, "b_pu": float, "line_MVA_1": float, "line_MVA_2": float, "line_MVA_3": float, 
               "control_bus": int, "side": int, "tap_ratio_final": float, "phase_shift": float, "tap_min": float, "tap_max": float, 
               "step_size": float, "min_V/MVA": float, "max_V/MVA": float}

def get_vals(data_row, data_locs, data_types): 
    data = {}
    for (name, val), (_, dtype) in zip(data_locs.items(), data_types.items()): 
        str_val = dtype(data_row[val[0] : val[1]].strip())
        data[name] = str_val
    return data

def get_item_dict(filename): 
    idx_bus_start = 2
    with open(filename) as f: 
        lines = []
        for line in f: 
            lines.append(line.rstrip())

    date_line = lines[0]
    S_base = float(date_line[30:38].strip())
    bus_data_info = lines[1]
    N_buses = int(bus_data_info[-14:].strip()[:-6]) # Experimental 

    bus_data = []
    for i, line in enumerate(lines[2:N_buses+2]):
        bus_data.append(get_vals(line, bus_locs, bus_types))

    idx = N_buses + 2 + 1 
    branch_data_info = lines[idx]
    N_branches = int(branch_data_info[-14:].strip()[:-6]) # Experimental 

    branch_data = []
    for i, branch in enumerate(lines[idx+1 : idx+1+N_branches]):
        branch_data.append(get_vals(branch, branch_locs, branch_types))

    # Collect data in one dict for bus and branch
    bus_final = {name: [] for name in bus_data[0]}
    for bus in bus_data: 
        for key in bus: 
            bus_final[key].append(bus[key])

    branch_final = {name: [] for name in branch_data[0]}
    for branch in branch_data: 
        for key in branch: 
            branch_final[key].append(branch[key])

    for name in bus_final: 
        bus_final[name] = np.array(bus_final[name])
    for name in branch_final: 
        branch_final[name] = np.array(branch_final[name])
    
    return bus_final, branch_final, S_base