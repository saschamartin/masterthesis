from utils import loadreference
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "./Data/2PD/JetA-Phasediagram.csv"
physpath = "./Data/physDatabank.csv"
    



with open("Data/2PD/JetA-Phasediagram.csv",'r') as f:
    f.seek(0)
    f.readline()

    p_boil = np.array([])
    t_boil = np.array([])
    p_dew = np.array([])
    t_dew = np.array([])
    while True:
        line = f.readline()
        if not line:
            break
        data_line = line.split(',')

        if data_line[2] == "LIQUID":
            p_boil = np.append(p_boil, np.array(data_line[1],dtype=float))
            t_boil = np.append(t_boil, np.array(data_line[0],dtype=float))
        elif data_line[2] == "VAPOR":
            p_dew = np.append(p_dew, np.array(data_line[1],dtype=float))
            t_dew = np.append(t_dew, np.array(data_line[0],dtype=float))
        else:
            print("No valid Phase input")
            break
    px_JetA = np.array([p_boil,t_boil]).T
    py_JetA =  np.array([p_dew,t_dew]).T


path_boil = "result_px.txt"
path_dew = "result_py.txt"
with open(path_boil,'r') as f:
    f.seek(0)
    f.readline()

    p_boil = np.array([])
    t_boil = np.array([])
    while True:
        data_line = f.readline().split()
        if not data_line:
            break
        p_boil = np.append(p_boil, np.array(data_line[0],dtype=float))
        t_boil = np.append(t_boil, np.array(data_line[1],dtype=float))

with open(path_dew,'r') as f:
    f.seek(0)
    f.readline()

    p_dew = np.array([])
    t_dew = np.array([])
    while True:
        data_line = f.readline().split()
        if not data_line:
            break
        p_dew = np.append(p_dew, np.array(data_line[0],dtype=float))
        t_dew = np.append(t_dew, np.array(data_line[1],dtype=float))


plt.figure(figsize=(8,6))
plt.plot(t_boil, p_boil, label='boiling curve')
plt.plot(t_dew, p_dew, label='dew curve')
plt.plot(px_JetA[:,1],px_JetA[:,0], label='expBoiling')
plt.plot(py_JetA[:,1],py_JetA[:,0], label='expDew')

plt.xlabel("Temperature [K]")
plt.ylabel("Pressure [atm]")

# xmin = min(min(t_boil), min(t_dew))
# xmax = max(max(t_boil), max(t_dew))
# ymin = min(min(p_boil), min(p_dew))
# ymax = max(max(p_boil), max(p_dew))
# xmarks = np.linspace(xmin,xmax,10)
# ymarks = np.linspace(ymin,ymax,10)

# plt.xticks(xmarks)
# plt.yticks(ymarks)

plt.grid(True)
plt.legend()
plt.show()



print("n")