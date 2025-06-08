import numpy as np
import matplotlib.pyplot as plt

def ajuste_da_reta(x, y, ylim=1e-16):
    below_x, below_y = [], []
    above_x, above_y = [], []
    quant = len(y)
    
    for i in range(quant - 1):
        x0, x1 = x[i], x[i+1]
        y0, y1 = y[i], y[i+1]
        
        # Caso 1: ambos abaixo ou igual ao limite
        if y0 <= ylim:
            below_x.append(x0)
            below_y.append(y0)
        else:
            above_x.append(x0)
            above_y.append(y0)
            
        if (y0 <= ylim) and (y1 >= ylim): 
            t = (ylim - y0) / (y1 - y0)
            x_cross = x0 + t * (x1 - x0)
            y_cross = ylim
            
            below_x.append(x_cross)
            below_y.append(y_cross)
            above_x.append(x_cross)
            above_y.append(y1)
        
        elif (y0 >= ylim) and (y1 <= ylim): 
            t = (ylim - y0) / (y1 - y0)
            x_cross = x0 + t * (x1 - x0)
            y_cross = ylim
            
            above_x.append(x_cross)
            above_y.append(y_cross)
            below_x.append(x_cross)
            below_y.append(y1)
            
        if (y0 >= ylim) and (y1 >= ylim): 
            below_x.append(x0)
            below_y.append(ylim)
            
        elif (y0 < ylim) and (y1 < ylim): 
            above_x.append(x0)
            above_y.append(ylim)
            
            
    return below_x, below_y, above_x, above_y


def count_switches(arr):
    switches = 0
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            switches += 1
    return switches

def graphic(values, labels, xlabel="", title = ""):
    fig, graf = plt.subplots(figsize = (6, 4), constrained_layout=True)
    for index, label in enumerate(labels): 
        graf.plot(sorted(values[index]), np.linspace(0, 1, len(values[index])), label=f"{label}")
    
    if xlabel == "SINR": 
        graf.set_xscale('log')
    graf.set(title=title, xlabel=f"{xlabel}")
    graf.grid(True, which='major', linestyle='-', linewidth=0.75)
    graf.grid(True, which='minor', linestyle=':', linewidth=0.5)
    graf.minorticks_on()    
    graf.legend()
    plt.show()