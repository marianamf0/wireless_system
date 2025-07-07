import numpy as np 
import matplotlib.pyplot as plt

def calculate_gain_and_associate_ap(position_UE, position_AP, X, R): 
    """
    Compute the channel gain matrix and associate each UE to the access point with the strongest path gain.

    Args:
        position_UE (ndarray): Complex array with UE positions in the plane.
        position_AP (ndarray): Complex array with AP positions in the plane.
        X (ndarray): Shadowing or fading matrix (UE x AP).
        R (ndarray): Fast fading matrix (UE x AP).

    Returns:
        Tuple[np.ndarray, List[int]]:
            - gain: Channel gain matrix (UE x AP).
            - associate_ap: List of AP indices associated with each UE (strongest path gain).
    """
    gain, associate_ap = [], []
    for index in range(len(position_UE)): 
        distance = abs(position_UE[index] - position_AP)

        aux_gain = [float(X[index, index_ap] * (1e-4/(max(distance[index_ap], 1)**(4))) * (R[index, index_ap]**2)) for index_ap in range(len(position_AP))]
        gain.append(aux_gain)
        
        path_gain = [X[index, index_ap] * (1e-4/(max(distance[index_ap], 1)**(4))) for index_ap in range(len(position_AP))]
        associate_ap.append(int(np.argmax(path_gain)))

    return np.array(gain), associate_ap

def get_scenario(type:str = "Noise-limited"): 
    """
    Generate the simulation scenario with positions, gains, and noise power.

    Args:
        type (str, optional): Scenario type.
            - "default": Returns a simple static gain configuration.
            - "Noise-limited": Returns a full setup with realistic gains and positions.
            - "Interference-limited": Scales positions to create higher interference conditions.

    Returns:
        Tuple:
            If type == "default":
                Tuple[np.ndarray, List[int], float]:
                    - gain: Static channel gain matrix.
                    - associate_ap: List of associated AP indices.
                    - noise_power: Noise power value.
            Otherwise:
                Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], float]:
                    - position_AP: Array of AP positions.
                    - position_UE: Array of UE positions.
                    - gain: Channel gain matrix.
                    - associate_ap: List of associated AP indices.
                    - noise_power: Noise power value.
    """
    if type == "default": 
        gain=np.array([[0.1], [0.05]]) 
        associate_ap=[0, 0]
        noise_power = 0.01
        return gain, associate_ap, noise_power
    
    position_AP = np.array([250 + 1j*750, 750 + 1j*750, 750 + 1j*250, 250 + 1j*250])
    position_UE = np.array([225.83 + 1j*203.33, 566.79 + 1j*321.88, 765.51 + 1j*146.88, 265.95 + 1j*702.39])
    
    if type == "Interference-limited":
        position_UE = position_UE/10
        position_AP = position_AP/10
    
    X = np.array([[5.3434e-2, 2.8731e-1, 1.9691e-2, 7.3013e-1],
                [3.2318,     1.5770,   2.6449e-1, 5.6379],
                [6.1470e-3, 1.1424,   2.6826e-1, 4.5709],
                [1.3485e-1, 4.6690e-1, 7.8250e-1, 1.6742]
                ])

    R = np.array([[1.248699, 3.248041, 0.772754, 0.708962],
                [0.498887, 0.104890, 0.647280, 0.940906],
                [0.382966, 0.682700, 1.891256, 0.327100],
                [0.065737, 0.649500, 1.981107, 1.259538]
                ])

    gain, associate_ap = calculate_gain_and_associate_ap(position_UE, position_AP, X, R)
    
    k0 = 1e-17*1e-3
    bandwidth, orthogonal_channels = 100e6, 1
    noise_power = k0*(bandwidth/orthogonal_channels)
    
    return position_AP, position_UE, gain, associate_ap, noise_power