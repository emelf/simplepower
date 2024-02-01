from typing import Optional, Sequence
from abc import ABC 
from enum import Enum
from ..utils import PQVD, BaseTimeSeries


class BusType(Enum): 
    PV = 0 
    PQ = 1 

    @staticmethod
    def get_type(bus_str: str) -> 'BusType': 
        if bus_str.upper() == "PQ":
            return BusType.PQ
        else:
            return BusType.PV
        

class BaseComponentModel(ABC):
    def __init__(self, bus_idx: int, bus_type: BusType, data: BaseTimeSeries): 
        self._bus_idx = bus_idx  
        self._bus_type = bus_type
        self._data = data

    @property
    def bus_idx(self): 
        return self._bus_idx

    @property
    def bus_type(self): 
        return self._bus_type
    
    @property
    def data(self): 
        return self._data

    def P_add_equation(self, pqvd: Optional[PQVD]=None, 
                       time_idx: Optional[int]=None) -> float: 
        return 0.0 

    def Q_add_equation(self, pqvd: Optional[PQVD]=None, 
                       time_idx: Optional[int]=None) -> float: 
        return 0.0 
    
    def V_add_equation(self, pqvd: Optional[PQVD]=None, 
                       time_idx: Optional[int]=None) -> float: 
        return 0.0 
    
