import numpy as np 
import matplotlib.pyplot as plt
from scenario import get_scenario

def calculate_sinr(gain, power, index_ue: int, index_ap: int, noise_power: float = 0.01): 
    """
    Calculate the SINR for a given user equipment (UE).

    Args:
        gain (ndarray): Channel gain matrix.
        power (ndarray): Array of transmit powers per UE.
        index_ue (int): Index of the target UE.
        index_ap (int): Index of the associated access point.
        noise_power (float, optional): Noise power level. Default is 0.01.

    Returns:
        float: SINR value for the specified UE.
    """
    interference = noise_power
    for index, p in enumerate(power):
        if index != index_ue: 
            interference += gain[index, index_ap]*p
        
    return gain[index_ue, index_ap]*power[index_ue]/interference

def power_control_dpc(scenario_type:str, target_sinr=1, pmax=1, pmin=1e-3, number_of_iterations:int=50): 
    """
    Perform Distributed Power Control (DPC) iterations to adjust transmit power towards the target SINR.

    Args:
        scenario_type (str): Scenario identifier ("default" or other).
        target_sinr (float, optional): Desired SINR target. Default is 1.
        pmax (float, optional): Maximum transmit power per UE. Default is 1 W.
        pmin (float, optional): Minimum transmit power per UE. Default is 1 mW.
        number_of_iterations (int, optional): Number of iterations to run. Default is 50.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Array with SINR values over iterations (iterations x UEs).
            - Array with power values over iterations (iterations x UEs).
    """
    
    if scenario_type == "default":
        number_ue = 2 
        gain, associate_ap, noise_power = get_scenario(type=scenario_type)
    else: 
        number_ue = 4
        _, _, gain, associate_ap, noise_power = get_scenario(type=scenario_type)
    
    power = np.array([pmax]*number_ue)
    value_sinr, value_power = [], [power]
    for _ in range(number_of_iterations): 
        sinr_ue = [calculate_sinr(gain=gain, power=value_power[-1], index_ue=index_ue, index_ap=associate_ap[index_ue], noise_power=noise_power) 
                   for index_ue in range(number_ue)]
        
        power_ue = [max(min(value_power[-1][index_ue] * target_sinr/sinr_ue[index_ue], pmax), pmin) for index_ue in range(number_ue)]
    
        value_sinr.append(sinr_ue)
        value_power.append(power_ue)
            
    return np.array(value_sinr), np.array(value_power)

def objective_function(gain, power, associate_ap, noise_power: float = 0.01, target_sinr=0.5): 
    """
    Compute the squared error objective function between actual and target SINR values.

    Args:
        gain (ndarray): Channel gain matrix.
        power (ndarray): Array of transmit powers per UE.
        associate_ap (list or ndarray): List of associated access points per UE.
        noise_power (float, optional): Noise power level. Default is 0.01.
        target_sinr (float, optional): Desired SINR target. Default is 0.5.

    Returns:
        float: Objective function value (sum of squared errors).
    """
    
    sinr_ue = [calculate_sinr(gain=gain, power=power, index_ue=index_ue, index_ap=associate_ap[index_ue], noise_power=noise_power) 
               for index_ue in range(len(power))]
    return sum((target_sinr - sinr)**2 for sinr in sinr_ue)

def objective_function_J1(gain, power, associate_ap, noise_power: float = 0.01): 
    """
    Compute the minimum SINR among all UEs.

    Args:
        gain (ndarray): Channel gain matrix.
        power (ndarray): Array of transmit powers per UE.
        associate_ap (list or ndarray): List of associated access points per UE.
        noise_power (float, optional): Noise power level. Default is 0.01.

    Returns:
        float: Minimum SINR across all UEs.
    """
    sinr_ue = [calculate_sinr(gain=gain, power=power, index_ue=index_ue, index_ap=associate_ap[index_ue], noise_power=noise_power) 
               for index_ue in range(len(power))]
    return min(sinr_ue)

def objective_function_J2(gain, power, associate_ap, noise_power: float = 0.01): 
    """
    Compute the sum of SINRs across all UEs.

    Args:
        gain (ndarray): Channel gain matrix.'
        power (ndarray): Array of transmit powers per UE.
        associate_ap (list or ndarray): List of associated access points per UE.
        noise_power (float, optional): Noise power level. Default is 0.01.

    Returns:
        float: Sum of SINRs.
    """
    sinr_ue = [calculate_sinr(gain=gain, power=power, index_ue=index_ue, index_ap=associate_ap[index_ue], noise_power=noise_power) 
               for index_ue in range(len(power))]
    return sum(sinr_ue)

def objective_function_J3(gain, power, associate_ap, noise_power: float = 0.01): 
    """
    Compute the product between the minimum SINR and the sum of SINRs across all UEs.

    Args:
        gain (ndarray): Channel gain matrix.
        power (ndarray): Array of transmit powers per UE.
        associate_ap (list or ndarray): List of associated access points per UE.
        noise_power (float, optional): Noise power level. Default is 0.01.

    Returns:
        float: Product of minimum SINR and sum of SINRs.
    """
    sinr_ue = [calculate_sinr(gain=gain, power=power, index_ue=index_ue, index_ap=associate_ap[index_ue], noise_power=noise_power) 
               for index_ue in range(len(power))]
    return min(sinr_ue)*sum(sinr_ue)


def control_power_fdm(scenario_type:str, target_sinr=1, pmax=1, pmin=1e-3, type_objective="default", number_of_iterations:int=200, eta=1e-2, mu=1e-2):
    """
    Perform power control using a gradient-based optimization approach for different objective functions.

    Args:
        scenario_type (str): Scenario identifier ("default" or other).
        target_sinr (float, optional): Desired SINR target. Default is 1.
        pmax (float, optional): Maximum transmit power per UE. Default is 1.
        pmin (float, optional): Minimum transmit power per UE. Default is 1e-3.
        type_objective (str, optional): Objective function type ("default", "J1", "J2", "J3"). Default is "default".
        number_of_iterations (int, optional): Number of iterations to run. Default is 200.
        eta (float, optional): Small perturbation for finite difference gradient estimation. Default is 1e-2.
        mu (float, optional): Learning rate for gradient updates. Default is 1e-2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Array with SINR values over iterations.
            - Array with power values over iterations.
    """
    
    if type_objective == "J1": 
        objective = objective_function_J1
    elif type_objective == "J2": 
        objective = objective_function_J2
    elif type_objective == "J3": 
        objective = objective_function_J3
    
    if scenario_type == "default": 
        number_ue = 2
        gain, associate_ap, noise_power = get_scenario(type=scenario_type)
    else: 
        number_ue = 4
        _, _, gain, associate_ap, noise_power = get_scenario(type=scenario_type)
    
    value_sinr, value_power = [], [np.array([pmax]*number_ue)]
    for k in range(number_of_iterations): 
        value_gradient = []
        if type_objective == "default": 
            value_objective_function = objective_function(gain=gain, power=value_power[-1], associate_ap=associate_ap, noise_power=noise_power, target_sinr=target_sinr)  
        else: 
            value_objective_function = objective(gain=gain, power=value_power[-1], associate_ap=associate_ap, noise_power=noise_power)
             
        for index_ue in range(number_ue): 
            power_copy = (value_power[-1]).copy()
            power_copy[index_ue] += eta
            
            if type_objective == "default": 
                new_value_objective_function = objective_function(gain=gain, power=power_copy, associate_ap=associate_ap, noise_power=noise_power, target_sinr=target_sinr) 
            else: 
                new_value_objective_function = objective(gain=gain, power=power_copy, associate_ap=associate_ap, noise_power=noise_power) 
            
            value_gradient.append((new_value_objective_function - value_objective_function)/eta)
        
        new_power = np.maximum(np.minimum(value_power[-1] - mu*np.array(value_gradient), pmax), pmin)
        value_power.append(new_power)
        
        new_sinr = [calculate_sinr(gain=gain, power=new_power, index_ue=index_ue, index_ap=associate_ap[index_ue], noise_power=noise_power) for index_ue in range(number_ue)]
        value_sinr.append(new_sinr)
    
    return np.array(value_sinr), np.array(value_power)

def graf_sinr_and_power(number_ue: int, value_sinr, value_power):
    """
    Plot SINR and power evolution over iterations for each UE.

    Args:
        number_ue (int): Number of user equipments (UEs).
        value_sinr (ndarray): Array with SINR values over iterations.
        value_power (ndarray): Array with power values over iterations.

    """
    fig, graf = plt.subplots(1, 2, figsize = (12, 4), constrained_layout=True)

    for index_ue in range(number_ue): 
        number_of_iterations = len(value_sinr[:, index_ue])
        graf[0].plot(list(range(number_of_iterations)), value_sinr[:, index_ue], label=f"$SINR_{index_ue + 1}$")
    
    graf[0].set(xlim=(0, number_of_iterations), xlabel="Number of iterations", ylabel="SINR")
    graf[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    graf[0].legend()
    graf[0].grid(True)

    for index_ue in range(number_ue): 
        number_of_iterations = len(value_power[:, index_ue])
        graf[1].plot(list(range(number_of_iterations)), value_power[:, index_ue], label=f"$P_{index_ue + 1}$")
        
    graf[1].set(xlim=(0, number_of_iterations+1), xlabel="Number of iterations", ylabel="Power")
    graf[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    graf[1].legend()
    graf[1].grid(True)
    
    plt.show()

def analysis(scenario_type:str, power: list, bandwidth: float = 100e6, orthogonal_channels=1): 
    """
    Compute and print the SINR, capacity, and energy efficiency per UE.

    Args:
        scenario_type (str): Scenario identifier ("default" or other).
        power (list or ndarray): List of transmit powers per UE.
        bandwidth (float, optional): Total system bandwidth in Hz. Default is 100e6.
        orthogonal_channels (int, optional): Number of orthogonal channels used. Default is 1.
    """
    if scenario_type == "default":
        number_ue = 2 
        gain, associate_ap, noise_power = get_scenario(type=scenario_type)
    else: 
        number_ue = 4
        _, _, gain, associate_ap, noise_power = get_scenario(type=scenario_type)
    
    sum_capacity = 0
    for index_ue in range(number_ue): 
        sinr = calculate_sinr(gain=gain, power=power, index_ue=index_ue, index_ap=associate_ap[index_ue], noise_power=noise_power)
        
        capacity = (bandwidth/orthogonal_channels)*np.log2(1 + sinr)
        print(f"\nPower for UE {index_ue+1}: {power[index_ue]}")
        print(f"SINR for UE {index_ue+1}:  {sinr:.2f}")
        print(f"Channel Capacity for UE {index_ue+1}: {(capacity/1e6):.2f} Mbps")
        
        sum_capacity += capacity
        
    print(f"\n\nSum Capacity: {(sum_capacity/1e6):.2f} Mbps")
    print(f"Energy Efficiency: {((sum_capacity/sum(power))/1e6):.2f} Mbits/J")