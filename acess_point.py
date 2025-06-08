import numpy as np

class AccessPoint: 
    
    def __init__(self, position, number_channels:int, bandwidth: float, number_ue: int, 
                 shadow_effect: bool = False, sigma_effect:float = 2.0):    
        
        self.index_ue = []
        self.position = position
        self.bandwidth = bandwidth 
        self.number_channels = number_channels
        self.shadow_coefficient = self.generate_shadow_coefficient(number_ue, shadow_effect, sigma_effect)
        
        self.k0 = 1e-17*1e-3 # Unit: W/Hz
        
    def generate_shadow_coefficient(self, number_ue: int, shadow_effect: bool = False, sigma_effect:float = 2.0): 
        if shadow_effect: 
            return np.exp(np.random.normal(0, sigma_effect, size=(number_ue)))
        return np.ones((number_ue))  
    
    def noise_power(self): 
        return self.k0*(self.bandwidth/self.number_channels)
    
    
        