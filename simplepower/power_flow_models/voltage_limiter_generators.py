import numpy as np 
from typing import Optional

from .base_models import BaseComponentModel
from ..utils import PQTimeSeries, PQVD, PowerFlowResult

class GeneratorVoltageLimiter(BaseComponentModel): 
    def __init__(self, ts: PQTimeSeries, bus_idx: int,
                 V_min: float, V_max: float, delta_Q: Optional[float]=1e7): 
        """
        Voltage limiter 
        --- 
        ts: The PQ timeseries that implements P0, Q0 in the following equations. 
        bus_idx: the bus where the PQ generator is located 
        V_min: Minimum allowed generator voltage. 
        V_max: Maximum allowed generator voltage 
        delta_S: The added reactive that is caused by a 1 pu voltage violation. 
        
        Equations
        ---
        Q_g = Q_ref + (max(V_min - V_g, 0) + min(V_g_max - V_g, 0))*delta_Q
        """
        super().__init__(bus_idx)
        self.ts = ts 
        self.V_min = V_min 
        self.V_max = V_max 
        self.delta_Q = delta_Q
    
    def dP_inj_equation(self, pqvd_vals: PQVD, ts: int):
        return 0.0
    
    def dQ_inj_equation(self, pqvd_vals: PQVD, ts: int):
        V = pqvd_vals.V_bus[self.bus_idx]
        dQ = (max(self.V_min - V, 0) + min(self.V_max - V, 0))*self.delta_Q
        return dQ
        

if __name__=="__main__": 
    pass 

