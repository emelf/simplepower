import numpy as np 
from typing import Optional

from .base_models import BaseComponentModel
from ..utils import PQTimeSeries, PQVD, PowerFlowResult

class ZIPLoadModel(BaseComponentModel): 
    def __init__(self, ts: PQTimeSeries, bus_idx: int,
                 a_p: Optional[float]=0.0, b_p: Optional[float]=0.0, c_p: Optional[float]=1.0, 
                 a_q: Optional[float]=0.0, b_q: Optional[float]=0.0, c_q: Optional[float]=1.0, 
                 V0_pu: Optional[float]=1.0): 
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
        super().__init__(bus_idx)
        self.ts = ts 
        self.a_p = a_p 
        self.b_p = b_p 
        self.c_p = c_p 
        self.a_q = a_q 
        self.b_q = b_q 
        self.c_q = c_q 
        self.V0 = V0_pu 
    
    def dP_inj_equation(self, pqvd_vals: PQVD, ts: int):
        V = pqvd_vals.V_bus[self.bus_idx]
        P0, Q0 = self.ts.iloc(ts) 
        dP = P0*(self.a_p*(V/self.V0)**2 + self.b_p*V/self.V0 + self.c_p)
        return dP 
    
    def dQ_inj_equation(self, pqvd_vals: PQVD, ts: int):
        V = pqvd_vals.V_bus[self.bus_idx]
        P0, Q0 = self.ts.iloc(ts) 
        dQ = Q0*(self.a_q*(V/self.V0)**2 + self.b_q*V/self.V0 + self.c_q)
        return dQ
        

if __name__=="__main__": 
    ts = PQTimeSeries(np.array([1.0]), np.array([1.0]))
    idx = 0 
    pf_res = PowerFlowResult(np.array([1.0]), # P
                             np.array([0.0]), # Q
                             np.array([1.0]), # V
                             np.array([0.0]), # delta
                             100.0, None)
    load_1 = ZIPLoadModel(ts, 0, 
                          a_p=0.0, b_p=1.0, c_p=0.0, 
                          a_q=0.0, b_q=1.0, c_q=0.0, 
                          V0_pu=1.0)
    

