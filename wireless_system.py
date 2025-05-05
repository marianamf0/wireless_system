import numpy as np
from acess_point import AccessPoint
from user_equipament import UserEquipament

class WirelessSystem: 
    
    def __init__(self, number_ap: int, number_ue:int,  orthogonal_channels:int = 1, size:int=1000):
        self.size = size
        self.orthogonal_channels = orthogonal_channels
        self.access_points = self.generate_access_points(number_ap)
        self.user_equipaments = self.generate_user_equipaments(number_ue)
        
        self.n = 4
        self.d0 = 1 # Unit: m 
        self.k = 1e-4 # Unit: m^4
        self.k0 = 1e-17*1e-3 # Unit: W/Hz
        self.bandwidth = 100 * 1e6 # Unit: Hz 
            
    def generate_access_points(self, number_ap: int = 1): 
        access_points = []
        quant = int(np.sqrt(number_ap))
        cell_width = self.size / quant
        for row in range(quant): 
            y = row * cell_width
            for col in range(quant): 
                x = col * cell_width
                position = (x + cell_width / 2,  y + cell_width / 2)
                access_points.append(AccessPoint(position, self.orthogonal_channels))
        
        return access_points
        
    def generate_user_equipaments(self, number_ue: int = 1): 
        return [UserEquipament(self.access_points, self.size) for _ in range(number_ue)]
    
    def power_received(self, user_equipament:UserEquipament, access_point: AccessPoint = None):
        if access_point is None: 
            access_point = self.access_points[user_equipament.index_ap]
            
        distance = user_equipament.calculate_distance(access_point.position)
        return user_equipament.power * (self.k/(max(distance, self.d0)**(self.n)))
    
    def noise_power(self): 
        return self.k0*(self.bandwidth/self.orthogonal_channels)
    
    def sum_power_per_channel(self, user_equipament:UserEquipament): 
        power = 0
        access_point = self.access_points[user_equipament.index_ap]
        
        for ue in self.user_equipaments: 
            if ue != user_equipament and ue.channel == user_equipament.channel:
                power += self.power_received(ue, access_point) 
                
        return power
    
    def calculate_SINR(self, user_equipament:UserEquipament): 
        power_received = self.power_received(user_equipament)
        sum_power_per_channel = self.sum_power_per_channel(user_equipament)
    
        return power_received/(sum_power_per_channel + self.noise_power())
    
    def channel_capacity(self, user_equipament:UserEquipament): 
        sinr = self.calculate_SINR(user_equipament)
        return (self.bandwidth/self.orthogonal_channels)*np.log2(1 + sinr)
    
    def evaluate(self, metric = "SINR"): 
        metric = self.calculate_SINR if metric == "SINR" else self.channel_capacity
        return [metric(ue) for ue in self.user_equipaments]