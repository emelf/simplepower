import numpy as np 
from typing import Generator, Optional
from enum import Enum

from .base_models import BaseComponentModel, BusType
from ..utils import BaseTimeSeries

class RandomPQVariable(BaseComponentModel): 
    """Used for either PQ or PV random variable in the grid. 
    
    Does not work!!
    ---
    
    """
    def __init__(self, bus_idx: int,
                 P_gen: Optional[Generator[float, None, None]]=None, 
                 Q_gen: Optional[Generator[float, None, None]]=None):
        """P_gen: A python generator object that yields a sample of the provided distribution.
        Q_gen: Same as the x_generator 
        """
        data = BaseTimeSeries(0.0, 0.0)
        bus_type = BusType.PQ
        super().__init__(bus_idx, bus_type, data)
        self.P_gen = P_gen 
        self.Q_gen = Q_gen 

    def P_add_equation(self, *args) -> float:
        return next(self.P_gen) if self.P_gen is not None else 0.0
    
    def Q_add_equation(self, *args) -> float:
        return next(self.Q_gen) if self.Q_gen is not None else 0.0