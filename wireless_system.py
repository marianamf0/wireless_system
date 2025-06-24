import random
import numpy as np
from acess_point import AccessPoint
from user_equipament import UserEquipament

def calculate_positions_ap(number_ap: int, size:int=1000): 
    position = []
    quant = int(np.sqrt(number_ap))
    cell_width = size / quant
    
    for row in range(quant): 
        y = row * cell_width
        for col in range(quant): 
            x = col * cell_width
            position.append((x + cell_width / 2,  y + cell_width / 2))
    
    return position

class WirelessSystem: 
    
    def __init__(self, number_ap: int, number_ue:int,  orthogonal_channels:int = 1, size:int=1000, shadow_effect: bool = False, 
                 random_channel:bool = True, channel_allocation:str = "", channel_aggregation: bool = False, power_control: bool = False, multipath_fading:bool = False):
        
        self.size = size
        self.bandwidth = 100e6 # Unit: Hz 
        self.orthogonal_channels = orthogonal_channels
        self.access_points = self.generate_access_points(number_ap, number_ue, shadow_effect, multipath_fading)
        self.user_equipaments = self.generate_user_equipaments(number_ue, random_channel, power_control)
        
        if not random_channel: 
            if channel_allocation == "PAPOA": 
                self.channel_allocation_PAPOA()
            elif channel_allocation == "IMCA": 
                self.channel_allocation_IMCA()
            else:
                self.channel_allocation()
            
        if channel_aggregation: 
            for ue in self.user_equipaments:
                ap = self.access_points[ue.index_ap]
                if len(ap.index_ue) == 1: 
                    ue.power = ue.power/self.orthogonal_channels
                    ue.channel_aggregation = True
        
            
    def generate_access_points(self, number_ap: int, number_ue: int, shadow_effect: bool = False, multipath_fading: bool = False, sigma_effect:float = 2.0): 
        position = calculate_positions_ap(number_ap=number_ap, size=self.size)
        return [AccessPoint(position=p, number_channels=self.orthogonal_channels, number_ue=number_ue, bandwidth= self.bandwidth,
                            shadow_effect=shadow_effect, sigma_effect=sigma_effect, multipath_fading=multipath_fading) for p in position]
          
    def generate_user_equipaments(self, number_ue: int, random_channel: bool, power_control: bool): 
        return [UserEquipament(access_points=self.access_points, size=self.size, index=index, 
                               random_channel=random_channel, power_control=power_control) for index in range(number_ue)]
    
    def channel_allocation(self): 
        channels = {i: 0 for i in range(1, self.orthogonal_channels + 1)}
        for ap in self.access_points: 
            available_channels = list(channels.keys())
            random.shuffle(available_channels)
            
            for index in ap.index_ue:
                if not available_channels:
                    break
                
                channel = available_channels.pop()
                self.user_equipaments[index].channel = channel
                channels[channel] += 1
        
        for ue in self.user_equipaments:
            if ue.channel is not None:
                continue
            
            least_used_channel = min(channels, key=channels.get)
            ue.channel = least_used_channel
            channels[least_used_channel] += 1
            
    def channel_allocation_PAPOA(self):
        for index, ue in enumerate(self.user_equipaments):
            if index < self.orthogonal_channels: 
                ue.channel = index + 1 
            else: 
                ap = self.access_points[ue.index_ap]  
                ue.channel = np.argmax(ap.multipath_coefficient[index]) + 1         
                
    def channel_allocation_IMCA(self): 
        for index, ue in enumerate(self.user_equipaments):
            access_point = self.access_points[ue.index_ap]
            noise = access_point.noise_power()
            noise_plus_interference = []
            for channel in range(self.orthogonal_channels):
                interference = self.sum_power_per_channel(user_equipament=ue, access_point=access_point, channel=channel+1)
                noise_plus_interference.append(interference+noise)
            
            ue.channel = np.argmin(noise_plus_interference) + 1    

    
    def sum_power_per_channel(self, user_equipament:UserEquipament, access_point: AccessPoint, channel:int = None): 
        power = 0        
        for ue in self.user_equipaments: 
            if ue.channel == None:
                continue
            if ue.index != user_equipament.index and (channel == ue.channel):
                power += ue.power_received(access_point) 
                
        return power
    
    def calculate_SINR(self, user_equipament:UserEquipament, channel:int): 
        access_point = self.access_points[user_equipament.index_ap]        
        power_received = user_equipament.power_received(access_point)
        interference = self.sum_power_per_channel(user_equipament, access_point, channel)
        sinr = power_received/(interference + access_point.noise_power())
        return sinr
    
    def channel_capacity(self, user_equipament:UserEquipament, channel:int): 
        sinr = self.calculate_SINR(user_equipament, channel)
        return (self.bandwidth/self.orthogonal_channels)*np.log2(1 + sinr)
    
    def evaluate(self, metric = "SINR"): 
        metric_func = self.calculate_SINR if metric == "SINR" else self.channel_capacity
        
        value_metrics = []
        for ue in self.user_equipaments: 
            if ue.channel_aggregation: 
                value = []
                for channel in range(self.orthogonal_channels): 
                    value += [metric_func(user_equipament=ue, channel=channel+1)]
                value_metrics += value if metric == "SINR" else [sum(value)]
            else: 
                value_metrics.append(metric_func(user_equipament=ue, channel=ue.channel))
            
        return value_metrics
        