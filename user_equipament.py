import random
import numpy as np
from acess_point import AccessPoint

class UserEquipament:
    
    def __init__(self, access_points:list[AccessPoint], size:int = 1000, power:float = 1):
        self.power = power
        self.position = self.generate_position(size)
        self.index_ap = self.get_index_to_acess_point(access_points)
        self.channel = self.generate_number_channel(access_points[self.index_ap])
            
    def generate_position(self, size):
        return (np.random.uniform(0, size), np.random.uniform(0, size))
    
    def get_index_to_acess_point(self, access_point: AccessPoint): 
        distance = [self.calculate_distance(ap.position) for ap in access_point]        
        return np.argmin(distance)
    
    def calculate_distance(self, position): 
        return np.sqrt((position[0] - self.position[0])**2 + (position[1] - self.position[1])**2)
    
    def generate_number_channel(self, access_point: AccessPoint): 
        return random.randint(1, access_point.number_channels)