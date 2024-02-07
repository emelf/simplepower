from typing import Optional, Sequence
from abc import ABC 
from enum import Enum
from ..common import PQVD, BaseTimeSeries
        

class BaseComponentModel(ABC):
    def __init__(self, bus_idx: int, data: BaseTimeSeries, 
                 level: Optional[int]=0): 
        """bus_idx: The PQ bus where the injections should be modified. 
        bus_type: PQ or PV bus. Only P and Q quantities works for now. 
        data: Time series data which will be used in the calculation of the additional P and Q injections. 
        level: The order of calculation. Lower values means earlier consideration in the power flow. Components in the same level will not have a specific calculation order. 
        """
        self._bus_idx = bus_idx  
        self.data = data
        self._level = level

    @property
    def bus_idx(self): 
        return self._bus_idx
    
    @property
    def level(self): 
        return self._level

    def P_add_equation(self, pqvd: Optional[PQVD]=None, 
                       time_idx: Optional[int]=None) -> float: 
        return 0.0 

    def Q_add_equation(self, pqvd: Optional[PQVD]=None, 
                       time_idx: Optional[int]=None) -> float: 
        return 0.0 
    
    def V_add_equation(self, pqvd: Optional[PQVD]=None, 
                       time_idx: Optional[int]=None) -> float: 
        return 0.0 
    
    def replace_data(self, new_data: BaseTimeSeries): 
        self.data = new_data
    
