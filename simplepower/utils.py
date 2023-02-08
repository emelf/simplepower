import numpy as np

class PowerFlowResult: 
    def __init__(self, P_calc, Q_calc, V_buses, d_buses, S_base): 
        self.P_calc = P_calc * S_base
        self.Q_calc = Q_calc * S_base
        self.V_buses = V_buses 
        self.d_buses = d_buses
        self.S_base = S_base

    def __repr__(self): 
        str1 = f"P_calc = {np.round(self.P_calc, 3)} MW \n" 
        str2 = f"Q_calc = {np.round(self.Q_calc, 3)} Mvar \n" 
        str3 = f"V_buses = {np.round(self.V_buses, 4)} pu \n" 
        str4 = f"d_buses = {np.round(self.d_buses*180/np.pi, 4)} deg \n" 
        return str1 + str2 + str3 + str4