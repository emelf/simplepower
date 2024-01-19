import numpy as np 
import pandas as pd 

class PQVD: 
    """ A simple way to store all bus P_inj, Q_inj, V_bus, d_bus values. """
    def __init__(self, P_bus, Q_bus, V_bus, d_bus): 
        self.P_bus = P_bus 
        self.Q_bus = Q_bus 
        self.V_bus = V_bus 
        self.d_bus = d_bus 

    def iloc(self, idx): 
        return (self.P_bus[idx], self.Q_bus[idx], self.V_bus[idx], self.d_bus[idx])

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