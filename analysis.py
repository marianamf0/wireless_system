import numpy as np
from wireless_system import WirelessSystem
from graphic import graphic, graphic_percentile, graphic_system

def run(number_ap:int, number_channel: int, number_ue: int = 1, iteration:int = int(1e4), shadow_effect:bool = False, random_channel: bool = True, channel_aggregation: bool = False, power_control: bool = False, verbose: bool = False): 
    sinr = []
    channel_capacity = []
    sum_capacity = []
    for _ in range(iteration): 
        system = WirelessSystem(
            number_ap=number_ap, number_ue=number_ue, orthogonal_channels=number_channel, 
            shadow_effect=shadow_effect, random_channel=random_channel, channel_aggregation=channel_aggregation, 
            power_control=power_control
        )
        
        sinr_value = system.evaluate()
        capacity_value = system.evaluate(metric="channel capacity")
        sinr = np.concatenate((sinr, sinr_value))
        channel_capacity = np.concatenate((channel_capacity, capacity_value))
        
        sum_capacity.append(sum(capacity_value))
        
        # if any(valor < 0.03 for valor in sinr_value): 
        #     value = system.evaluate(verbose=True)
        #     graphic_system(system)
        #     break
    
    if verbose: 
        print(f"Para M = {number_ap}, K = {number_ue}, N = {number_channel}:")
        print(f"The 10th percentile of the SINR: {np.percentile(sinr, 10)}")
        print(f"The 50th percentile of the SINR: {np.percentile(sinr, 50)}")
        print(f"The 10th percentile of the Capacity Channel: {np.percentile(channel_capacity, 10)/1e6} Mbps")
        print(f"The 50th percentile of the Capacity Channel: {np.percentile(channel_capacity, 50)/1e6} Mbps")
        print(f"Average sum-capacity to {number_ap} AP: {(np.mean(sum_capacity)/ 1e6):.2f} Mbps")
        print(f"Spectrum efficiency to {number_ap} AP: {(np.mean(sum_capacity)/(100 * 1e6 * 1)):.2f} bits/second/Hz/km²")
        
        #spectrum_efficiency = [value/(100 * 1e6 * 1) for value in sum_capacity]
        
    return sinr, channel_capacity

def coverage(number_ue: int = 1, list_acess_point: list = [1, 9, 36, 64]): 
    output = []
    for M in list_acess_point: 
        for N in range(1, 4): 
            sinr, channel_capacity = run(number_ap=M, number_channel=N, number_ue=number_ue)
            
            output.append({
                "Channel": N,
                "Access points": M,
                "SINR": sinr, 
                "Channel Capacity": [c / 1e6 for c in channel_capacity]
            })
            
    return output

if __name__ == "__main__": 
    
    # Exercise 9 
    print("\n K = 1")
    output = coverage(number_ue=1)
    graphic(output, parameter="Access points", metric="SINR", name="image1_k1_SINR_access_points")
    graphic(output, parameter="Access points", metric="Channel Capacity", name="image2_k1_capacity_access_points")
    graphic(output, parameter="Channel", metric="SINR", name="image3_k1_SINR_channel")
    graphic(output, parameter="Channel", metric="Channel Capacity", name="image4_k1_capacity_channel")

    graphic_percentile(output, parameter="Channel", metric="SINR", name="image5_k1_percentile_SINR_channel")
    graphic_percentile(output, parameter="Channel", metric="Channel Capacity", name="image6_k1_percentile_capacity_channel")
    graphic_percentile(output, parameter="Access points", metric="SINR", name="image7_k1_percentile_SINR_access_points")
    graphic_percentile(output, parameter="Access points", metric="Channel Capacity", name="image8_k1_percentile_capacity_access_points")
    
    # Exercise 10
    print("\n K = 13")
    output = coverage(number_ue=13)
    graphic(output, parameter="Access points", metric="SINR", name="image9_k13_SINR_access_points")
    graphic(output, parameter="Access points", metric="Channel Capacity", name="image10_k13_capacity_access_points")
    graphic(output, parameter="Channel", metric="SINR", name="image11_k13_SINR_channel")
    graphic(output, parameter="Channel", metric="Channel Capacity", name="image12_k13_capacity_channel")

    graphic_percentile(output, parameter="Channel", metric="SINR", name="image13_k13_percentile_SINR_channel")
    graphic_percentile(output, parameter="Channel", metric="Channel Capacity", name="image14_k13_percentile_capacity_channel")
    graphic_percentile(output, parameter="Access points", metric="SINR", name="image15_k13_percentile_SINR_access_points")
    graphic_percentile(output, parameter="Access points", metric="Channel Capacity", name="image16_k13_percentile_capacity_access_points")
