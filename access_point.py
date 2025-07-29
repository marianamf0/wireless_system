import numpy as np

class AccessPoint: 
    
    def __init__(self, position, number_channels:int, bandwidth: float, number_ue: int, 
                 shadow_effect: bool = False, sigma_effect:float = 2.0, multipath_fading: bool = False):    
        
        self.index_ue = []
        self.position = position
        self.bandwidth = bandwidth 
        self.number_channels = number_channels
        #self.channels_per_ue = {i: 0 for i in range(1, self.number_channels + 1)}
        self.shadow_coefficient = self.generate_shadow_coefficient(number_ue, shadow_effect, sigma_effect)
        self.multipath_coefficient = self.generate_multipath_fading(multipath_fading, number_ue, number_channels)
        self.k0 = 1e-17*1e-3 # Unit: W/Hz
        
    def generate_multipath_fading(self, multipath_fading:bool, number_ue:int, number_channels:int, sigma_fading: float = 1/np.sqrt(2)): 
        if multipath_fading: 
            x1 = np.random.normal(0, sigma_fading, (number_ue, number_channels))
            x2 = np.random.normal(0, sigma_fading, (number_ue, number_channels))
            
            return np.sqrt(x1**2 + x2**2)
        
        return np.ones((number_ue, number_channels))
        
    def generate_shadow_coefficient(self, number_ue: int, shadow_effect: bool = False, sigma_effect:float = 2.0): 
        if shadow_effect: 
            return np.exp(np.random.normal(0, sigma_effect, size=(number_ue)))
        return np.ones((number_ue))  
    
    def noise_power(self): 
        return self.k0*(self.bandwidth/self.number_channels)
    
    
        