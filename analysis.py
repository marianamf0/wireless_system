from wireless_system import WirelessSystem
from graphic import graphic, graphic_percentile, graphic_system

def run(number_ap:int, number_channel: int, number_ue: int = 1, iteration:int = 10e4): 
    sinr = []
    channel_capacity = []
    
    for _ in range(iteration + 1): 
        system = WirelessSystem(
            number_ap=number_ap, number_ue=number_ue, orthogonal_channels=number_channel
        )
        
        sinr += system.evaluate()
        channel_capacity += system.evaluate(metric="channel capacity")
        
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
    
    # Exercise - K = 14
    print("\n K = 14")
    output = coverage(number_ue=14)
    graphic(output, parameter="Access points", metric="SINR", name="image17_k14_SINR_access_points")
    graphic(output, parameter="Access points", metric="Channel Capacity", name="image18_k14_capacity_access_points")
    graphic(output, parameter="Channel", metric="SINR", name="image19_k14_SINR_channel")
    graphic(output, parameter="Channel", metric="Channel Capacity", name="image20_k14_capacity_channel")

    graphic_percentile(output, parameter="Channel", metric="SINR", name="image21_k14_percentile_SINR_channel")
    graphic_percentile(output, parameter="Channel", metric="Channel Capacity", name="image22_k14_percentile_capacity_channel")
    graphic_percentile(output, parameter="Access points", metric="SINR", name="image23_k14_percentile_SINR_access_points")
    graphic_percentile(output, parameter="Access points", metric="Channel Capacity", name="image24_k14_percentile_capacity_access_points")
    
