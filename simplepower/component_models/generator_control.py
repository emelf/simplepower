import numpy as np 
from typing import Optional

from simplepower.utils import PQVD

from .base_models import BaseComponentModel 
from ..utils import BaseTimeSeries, PQVD 


class GeneratorVDroop(BaseComponentModel): 
    def __init__(self, bus_idx: int,
                 V_ref: float, Q_ref: float, k_droop: float, 
                 Q_ref_pull: Optional[float]=1e1): 
        """
        Voltage Droop for Static genereator models (PQ model)
        --- 
        bus_idx: the bus where the PQ generator is located 
        V_ref: The voltage that causes Q_ref reactive power 
        Q_ref: The reactive power dispatch at V_ref 
        k_droop: The slope of between V and Q
        
        Equations
        ---
        dV = V - V_ref 
        dQ = (Q - Q_ref) + dV*k_droop 
        """
        data = BaseTimeSeries(0.0, 0.0)
        super().__init__(bus_idx, data)
        self.V_ref = V_ref 
        self.Q_ref = Q_ref
        self.k_droop = k_droop 
        self.Q_ref_pull = Q_ref_pull
        
    def P_add_equation(self, pqvd: PQVD, ts: int):
        return 0.0
    
    def Q_add_equation(self, pqvd: PQVD, ts: int):
        P, Q, V, delta = pqvd.iloc_mva(self.bus_idx)
        dV = V - self.V_ref 
        dQ = (self.Q_ref - Q)*self.Q_ref_pull + dV*self.k_droop 
        return dQ
    
    
class GeneratorVoltageLimiter(BaseComponentModel): 
    def __init__(self, bus_idx: int, limiter_data: BaseTimeSeries,
                 delta_Q: float = 1e5): 
        """
        Voltage limiter
        --- 
        bus_idx: the bus where the PQ generator is located 
        limiter_data: The timeseries data for V_min, V_max
        delta_Q: The added reactive that is caused by a 1 pu voltage violation. 
        
        Equations
        ---
        Q_g = Q_ref + (max(V_min - V_g, 0) + min(V_g_max - V_g, 0))*delta_Q
        """
        super().__init__(bus_idx, limiter_data)
        self.delta_Q = delta_Q
    
    def P_add_equation(self, pqvd: PQVD, time_idx: int):
        return 0.0
    
    def Q_add_equation(self, pqvd: PQVD, time_idx: int):
        V_min, V_max = self.data.iloc(time_idx)
        P, Q, V, delta = pqvd.iloc_mva(self.bus_idx)
        dQ = (max(V_min - V, 0) + min(V_max - V, 0))*self.delta_Q
        return dQ
    

class GeneratorVControlPQ(BaseComponentModel): 
    def __init__(self, bus_idx: int, V_ref_data: BaseTimeSeries,
                 delta_Q: Optional[float]=1e7): 
        """
        Voltage limiter
        --- 
        bus_idx: the bus where the PQ generator is located 
        V_ref_data: The timeseries that implements (V_ref, None) in the following equations. 
        delta_Q: The added reactive that is caused by a 1 pu voltage violation. 
        
        Equations
        ---
        Q_g = Q_ref + (max(V_min - V_g, 0) + min(V_g_max - V_g, 0))*delta_Q
        """
        super().__init__(bus_idx, V_ref_data)
        self.delta_Q = delta_Q
    
    def P_add_equation(self, pqvd: PQVD, time_idx: int):
        return 0.0
    
    def Q_add_equation(self, pqvd: PQVD, time_idx: int):
        V_ref, _ = self.data.iloc(time_idx)
        P, Q, V, delta = pqvd.iloc_mva(self.bus_idx)
        dQ = (V_ref - V)*self.delta_Q
        return dQ
    

class DistributedPSlack(BaseComponentModel): 
    def __init__(self, bus_idx: int, slack_bus_idx: int, weight: float = 100.0): 
        """
        Distributed Slack Component
        --- 

        Forces the slack bus to produce close to 0 active power
        bus_idx: the bus where the generator is located 
        slack_bus_idx: the bus index where the slack bus is located 
        weight: A high value means  [0, 1] 
        
        Equations
        ---
        P_g_add = P_slack * share + P_add_integral
        """
        super().__init__(bus_idx, None, level=-1) 
        self.weight = weight 
        self.slack_bus_idx = slack_bus_idx

    def P_add_equation(self, pqvd: PQVD, time_idx: int) -> float:
        P_slack, _, _, _ = pqvd.iloc_mva(self.slack_bus_idx)
        P_new = P_slack * self.weight
        return P_new