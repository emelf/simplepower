import numpy as np 
from typing import Optional

from simplepower.utils import PQVD

from .base_models import BaseComponentModel, BusType
from ..utils import BaseTimeSeries, PQVD 


class BasePQGenerator(BaseComponentModel): 
    def __init__(self, bus_idx: int, PQ_data: BaseTimeSeries): 
        """
        Base generator class 
        --- 
        Used for adding the power reference to a generator with controls 
        """
        bus_type = BusType.PQ
        super().__init__(bus_idx, bus_type, PQ_data)
    
    def P_add_equation(self, pqvd: PQVD, time_idx: int) -> float:
        P0, Q0 = self.data.iloc(time_idx) 
        return P0
    
    def Q_add_equation(self, pqvd: PQVD, time_idx: int) -> float:
        P0, Q0 = self.data.iloc(time_idx) 
        return Q0
    

class BasePVGenerator(BaseComponentModel): 
    def __init__(self, bus_idx: int, PV_data: BaseTimeSeries): 
        """
        Base generator class 
        --- 
        Used for adding the power reference to a generator with controls 
        """
        bus_type = BusType.PV
        super().__init__(bus_idx, bus_type, PV_data)
    
    def P_add_equation(self, pqvd: PQVD, time_idx: int) -> float:
        P0, V0 = self.data.iloc(time_idx) 
        return P0
    
    def V_add_equation(self, pqvd: PQVD, time_idx: int) -> float:
        P0, V0 = self.data.iloc(time_idx) 
        return V0
        

if __name__=="__main__": 
    pass 

