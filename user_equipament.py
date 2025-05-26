import random
import numpy as np
from acess_point import AccessPoint

class UserEquipament:
    
    def __init__(self, access_points:list[AccessPoint], random_channel: bool, power_control: bool, size:int = 1000, power:float = 1, index: int = 0):
        self.n = 4
        self.d0 = 1 # Unit: m 
        self.k = 1e-4 # Unit: m^4
        
        self.index = index
        self.channel_aggregation = False
        self.position = self.generate_position(size)
        self.index_ap = self.get_index_to_acess_point(access_points)
        access_points[self.index_ap].index_ue.append(self.index)
        self.channel = self.generate_number_channel(access_points[self.index_ap]) if random_channel else None
        self.power = self.calculate_power(access_points[self.index_ap]) if power_control else power
        
    def generate_position(self, size):
        return (np.random.uniform(0, size), np.random.uniform(0, size))
    
    def get_index_to_acess_point(self, access_point: AccessPoint, define_distance: bool = True):
        if define_distance: 
            return np.argmin([self.calculate_distance(ap.position) for ap in access_point])  
        print("get_index_to_acess_point", "define_distance", define_distance)
        return np.argmax([self.path_gain(ap) for ap in access_point])
    
    def calculate_distance(self, position): 
        return np.sqrt((position[0] - self.position[0])**2 + (position[1] - self.position[1])**2)
    
    def generate_number_channel(self, access_point: AccessPoint): 
        return random.randint(1, access_point.number_channels)
    
    def calculate_power(self, access_point: AccessPoint): 
        print("calculate_power")
        power = access_point.noise_power() / self.path_gain(access_point)
        return min(power, 1)
    
    def power_received(self, access_point: AccessPoint): 
        #print("power_received", self.power)
        distance = self.calculate_distance(access_point.position)
        shadow_coefficient = access_point.shadow_coefficient[self.index]     
        shadow_coefficient = 1
        return self.power * shadow_coefficient * (self.k/(max(distance, self.d0)**(self.n)))
    
    def path_gain(self, access_point: AccessPoint):     
        distance = self.calculate_distance(access_point.position)
        shadow_coefficient = access_point.shadow_coefficient[self.index] 
        
        return shadow_coefficient * (self.k/(max(distance, self.d0)**(self.n)))