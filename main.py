import sys
sys.path.insert(0, "./utensils")
sys.path.append("./phase/DistillationCurve/distillation")
import pandas as pd
from algorithms import NSGA_algo
from deap import base, tools, algorithms
from utensils.setup import (loadreference, validate_inputs)


def optimizer(
    filename: str,
    number_components: int = None,
    components: list = None,
    ratios: list = None,
    linear_props: list = None,
    target_fuel_properties: dict = None,
    group_map: dict = None,
    group_limits: dict = None):

    validate_inputs("NSGA_III",physicalDatabank_path,number_components,components,ratios,linear_props,target_fuel_properties, group_map, group_limits)
    df = pd.read_csv(physicalDatabank_path)

    return NSGA_algo(
        max_blend_size=number_components,  # wird ignoriert bei fixed_components
        df=df,
        target_fuel_properties=target_fuel_properties,
        group_map=group_map,
        group_limits=group_limits,
        POP_SIZE=28,
        NGEN=500,
        ELITE_SIZE=5,
        early_stop_gen=100,
        fixed_components=components  # Fixe Components
    )

def userinput(physicalDatabank_path,group_map):
    invertet_map = {val: key for key, values in group_map.items() for val in values}
    group_limits = {
        "Paraffin": (0.0,0.0),
        "Naphthene": (0.0,0.0),
        "Aromatic": (0.0,0.0),
        "Oxygenate": (0.0, 0.0),
        }
    df = pd.read_csv(physicalDatabank_path)
    number_components = 0
    fixed = 0
    components = []
    required = []

    print("=================================")
    print("           User Input            ")
    print("Please input the number of components in the mixture.")
    number_components = int(input("> "))
    k = list(group_limits.keys())
    for i in range(len(k)):
        print(f"------- {k[i]} -------")
        while True:
            try:
                l = float(input("lower limit [0,1]: "))
            except:
                print("Please input a numerical value")
                continue

            if l > 1.0 or l < 0.0:
                print("Please choose a value in the interval [0;1]")
            else:
                break

        while True:
            try:
                u = float(input(f"upper limit [{l},1]: "))
            except:
                print("Please input a numerical value")
                continue
            if u < l:
                print(f"Please choose a value equal or bigger than the lower limit of {l}")
            elif l > 1.0 or l < 0.0:
                print("Please choose a value in the interval [0;1]")
            elif l == "":
                print("Please input a numerical value")
            else:
                if u > 0.0:
                    required.append(k[i])
                break
        scope = (l,u)
        print(f"chosen range: {scope}")
        group_limits[k[i]] = scope 

    while True:
        print("Define the number of fixed componentes.")
        fixed = int(input("> "))
        if fixed == 0:
            components = None
            return number_components, components, group_limits
        elif fixed < len(required):
            print("Please add more fixed components")
        else:
            break


    print("Choose a component by writing the name from the following list:")
    print(df.to_string(index=False))
    while True:
        groups = []
        components = []
        for n in range(fixed):
            while True:
                comp = input("> ")
                hit = df[df["name"] == comp]
                if hit.empty:
                    print(f"Name '{hit}' not found")
                elif comp in components:
                    print("Please choose another component.")
                else:
                    group = hit.iloc[0]["family"]
                    group = invertet_map.get(group)
                    groups.append(group)
                    components.append(comp)
                    break

        missing = [k for k in required if k not in groups]
        if missing:
            continue
        else: 
            break

    return number_components, components, group_limits

if __name__ == "__main__":
    physicalDatabank_path = "Data/physDatabank.csv"
    referenceFuelPD_path = "Data/2PD/JetA-Phasediagram.csv"
    expDistCurve = "Data/2PD/distillationcurve.csv"
    expPDCurve = "Data/2PD/JetA-Phasediagram.csv"
    
    target_fuel_properties, group_map, _ = loadreference(expDistCurve,expPDCurve)
    number_components, components, group_limits = userinput(physicalDatabank_path, group_map)

    # number_components = 5
    # components = ["n-Propyl-Cyclohexane","Iso-Octane","n-Dodecane","1-Methyl-Naphthalene","n-Hexadecane"]
    ratios = None
    linear_props = None
    
   

    try:
        optimizer(
        filename=physicalDatabank_path,
        number_components=number_components,
        components=components,
        ratios=ratios,
        linear_props=linear_props,
        target_fuel_properties=target_fuel_properties,
        group_map=group_map, 
        group_limits=group_limits,
        )
    except ValueError as e:
        import traceback,sys
        print("❌ Error:", e)
        for frame in traceback.extract_tb(sys.exc_info()[2]):
            fname,lineno,fn,text = frame
            print("Error in %s on line %d" % (fname, lineno))
    except Exception as e:
        import traceback,sys
        print("❌ Unexpected error:", e)
        for frame in traceback.extract_tb(sys.exc_info()[2]):
            fname,lineno,fn,text = frame
            print("Error in %s on line %d" % (fname, lineno)) 
        
