import numpy as np 
import pandas as pd 
from abc import ABC, abstractmethod 
from ..utils import PQTimeSeries, PQVD


class BaseComponentModel(ABC): 
    def __init__(self, bus_idx, *args, **kwargs): 
        self.bus_idx = bus_idx  

    @abstractmethod
    def dP_inj_equation(self, pqvd_vals: PQVD, ts: int): 
        """Explenation: This method is the term that is on the right hand side of 
        the following equation: 
        P_calc_i = P_known_i -> Uses this relation to find the root P_calc_i - P_known_i = 0. 
        P_known_i is usually constant for static generators. However, we can introduce dynamic behavior 
        by allowing the "known" value to be dependend on the solution itself. This is e.g., voltage droop control. """
        return pqvd_vals.iloc[self.bus_idx][0] # Default: return the same value 

    @abstractmethod
    def dQ_inj_equation(self, pqvd_vals: PQVD, ts: int): 
        return pqvd_vals.iloc[self.bus_idx][1] # Default: return the same value 
    
class BaseGenerator(BaseComponentModel): 
    def __init__(self, ts: PQTimeSeries, bus_idx: int): 
        """
        Base generator class 
        --- 
        Used for adding the power reference to a generator with controls 
        """
        super().__init__(bus_idx)
        self.ts = ts 
    
    def dP_inj_equation(self, pqvd_vals: PQVD, ts: int):
        P0, Q0 = self.ts.iloc(ts) 
        return P0
    
    def dQ_inj_equation(self, pqvd_vals: PQVD, ts: int):
        P0, Q0 = self.ts.iloc(ts) 
        return Q0
    
    
class BaseLoad(BaseComponentModel): 
    def __init__(self, ts: PQTimeSeries, bus_idx: int): 
        """
        Base load class 
        --- 
        Used for adding the power reference to a load with controls
        """
        super().__init__(bus_idx)
        self.ts = ts 
    
    def dP_inj_equation(self, pqvd_vals: PQVD, ts: int):
        P0, Q0 = self.ts.iloc(ts) 
        return P0
    
    def dQ_inj_equation(self, pqvd_vals: PQVD, ts: int):
        P0, Q0 = self.ts.iloc(ts) 
        return Q0
 