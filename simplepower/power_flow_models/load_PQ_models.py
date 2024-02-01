import numpy as np 
from typing import Optional

from .base_models import BaseComponentModel, BusType 
from ..utils import BaseTimeSeries, PQVD 

class BasePQLoad(BaseComponentModel): 
    def __init__(self, bus_idx: int, PQ_data: BaseTimeSeries): 
        """
        Base load class 
        --- 
        Used for adding the power reference to a load with controls
        """
        bus_type = BusType.PQ
        super().__init__(bus_idx, bus_type, PQ_data)
    
    def P_add_equation(self, pqvd: PQVD, time_idx: int):
        P0, Q0 = self.data.iloc(time_idx) 
        return P0
    
    def Q_add_equation(self, pqvd: PQVD, time_idx: int):
        P0, Q0 = self.data.iloc(time_idx) 
        return Q0
    

class ZIPLoadModel(BaseComponentModel): 
    def __init__(self, bus_idx: int, load_data: BaseTimeSeries,
                 a_p: float = 0.0, b_p: float = 0.0, c_p: float = 1.0, 
                 a_q: float = 0.0, b_q: float = 0.0, c_q: float = 1.0, 
                 V0_pu: float = 1.0): 
        """
        ZipLoadModel 
        --- 

        ts: The PQ timeseries that implements P0, Q0 in the following equations. 
        a_p, b_p, c_p: Parameters for the active power behavior 
        a_q, b_q, c_q: Parameters for the reactive power behavior 
        ---
        P = P0(a_p*(V/V0)^2 + b_p*(V/V0) + c_p) \n
        Q = Q0(a_q*(V/V0)^2 + b_q*(V/V0) + c_q) 
        """
        bus_type = BusType.PQ
        super().__init__(bus_idx, bus_type, load_data)

        self.a_p = a_p 
        self.b_p = b_p 
        self.c_p = c_p 
        self.a_q = a_q 
        self.b_q = b_q 
        self.c_q = c_q 
        self.V0 = V0_pu 
    
    def P_add_equation(self, pqvd: PQVD, time_idx: int):
        V = pqvd.V_bus[self.bus_idx]
        P0, Q0 = self.data.iloc(time_idx) 
        dP = P0*(self.a_p*(V/self.V0)**2 + self.b_p*V/self.V0 + self.c_p - 1.0)
        return dP 
    
    def Q_add_equation(self, pqvd: PQVD, time_idx: int):
        V = pqvd.V_bus[self.bus_idx]
        P0, Q0 = self.data.iloc(time_idx) 
        dQ = Q0*(self.a_q*(V/self.V0)**2 + self.b_q*V/self.V0 + self.c_q - 1.0)
        return dQ
        

if __name__=="__main__": 
    pass 
    

