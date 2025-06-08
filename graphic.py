import numpy as np
import matplotlib.pyplot as plt
from wireless_system import WirelessSystem
from matplotlib.ticker import ScalarFormatter

def graphic_system(system: WirelessSystem): 
        user_equipaments, access_points = system.user_equipaments, system.access_points
        list_color = ['black', 'violet', 'green']
        
        x = [ap.position[0] for ap in access_points]
        y = [ap.position[1] for ap in access_points]
        
        index_ap = [ue.index_ap for ue in user_equipaments]
        x_ue = [ue.position[0] for ue in user_equipaments]
        y_ue = [ue.position[1] for ue in user_equipaments]

        fig, graf = plt.subplots(figsize = (5, 4))
        graf.scatter(x, y, marker='^', s=100, facecolors='none', edgecolors='blue')
        graf.scatter(x_ue, y_ue, marker='s', s=8, facecolors='none', edgecolors='red')
        
        line_ap = np.linspace(0, system.size, int(np.sqrt(len(access_points)) + 1))
        for value in line_ap: 
            graf.axvline(x=value, color='gray', linestyle='--')
            graf.axhline(y=value, color='gray', linestyle='--')
 
        for index, ap in enumerate(index_ap):
            index_color = user_equipaments[index].channel
            graf.plot([x_ue[index], x[ap]], [y_ue[index], y[ap]], linestyle='-', color=list_color[index_color-1], linewidth=1)
            
        graf.set_xlim(0, system.size)
        graf.set_ylim(0, system.size)
        
        fig.savefig("image/example_wireless_system.png", bbox_inches='tight', pad_inches=0)
          
        plt.show()

def graphic(output, parameter = "Access points", metric = "SINR", name = "image"): 
    list_values = sorted(list(set([dict[parameter] for dict in output])))
    parameter_line = "Channel" if parameter == "Access points" else "Access points"
    title_line = "M" if parameter_line == "Access points" else "N"
    title_xlabel = metric + " (Mbps)" if metric == "Channel Capacity" else metric
    
    fig, graf = plt.subplots(1, len(list_values), figsize = (15, 7.5))
    
    for dict in output: 
        index = list_values.index(dict[parameter])
        value = dict[metric]
        
        graf[index].plot(sorted(value), np.linspace(0, 1, len(value)), label=f"{title_line}: {dict[parameter_line]}")
        
        if metric == "Channel Capacity" and np.percentile(value, 10) >= 100:
            print(f"{parameter}: {dict[parameter]}")
            print(f"{title_line}: {dict[parameter_line]}") 
            print(np.percentile(value, 10))
            
       
    for index, value in enumerate(list_values): 
        if metric == "Channel Capacity": 
            graf[index].axvline(x=100, color='black', linestyle='--')
        else: 
            graf[index].set_xscale('log')
        
        graf[index].set(title=f"{parameter}: {value}", xlabel = title_xlabel)
        graf[index].grid(True, which='major', linestyle='-', linewidth=0.75)
        graf[index].grid(True, which='minor', linestyle=':', linewidth=0.5)
        graf[index].minorticks_on()
        graf[index].legend()
    
    fig.savefig(f"image/{name}.png", bbox_inches='tight', pad_inches=0)
    
    plt.close(fig)
    
def search(data, number_access_points:int, number_channel: int): 
    for element in data: 
        if element["Channel"] == number_channel and element["Access points"] == number_access_points: 
            return element
    return None
    
def graphic_percentile(output, parameter = "Access points", metric = "SINR", name="image"): 
    list_values = sorted(list(set([dict[parameter] for dict in output])))
    title_line = "M" if parameter == "Access points" else "N"
    parameter_line = "Channel" if parameter == "Access points" else "Access points"
    list_values_line = sorted(list(set([dict[parameter_line] for dict in output])))
    title_ylabel = metric + " (Mbps)" if metric == "Channel Capacity" else metric
    
    fig, graf = plt.subplots(figsize = (6, 4), constrained_layout=True)
    
    for value in list_values: 
        x_value = []
        y_value = []
        for line in list_values_line: 
            output_data = search(
                data=output, 
                number_access_points= value if parameter == "Access points" else line, 
                number_channel= value if parameter == "Channel" else line
            )
            x_value.append(line)
            y_value.append(np.percentile(output_data[metric], 10)) 
            
        graf.plot(x_value, y_value, label=f"{title_line}: {value}", marker='o')
    
    graf.set_xticks(list_values_line)
    graf.set(title=" ", xlabel = parameter_line, ylabel=title_ylabel)
    graf.grid(True)
    graf.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    fig.savefig(f"image/{name}.png", bbox_inches='tight', pad_inches=0)
    
    plt.close(fig)