import sys
import numpy as np
from numpy import random
import random
from sklearn.neighbors import NearestNeighbors
# from properties import calculate_mixture_properties
# sys.path.append("C:/Users/sp-ma/Documents/Studium/#Masterarbeit/Code/phase/DistillationCurve/distillation")

def loadreference(path_dist,path_PD):
    '''
    loading of the relevant reference values and relevant groups
    
    Args:
        path_dist: relative path to distillation curve file
        path_PD: relative path to Phase diagram file

    Returns: 
        target_fuel_properties: dict with reference values
        group_map: dict with groups
        group_limits: dict with the determined grouplimits
    '''
    target_fuel_properties = {
        "density": 738.99,
        "molar_mass": 144.36,
        "viscosity": 2.88,
        "TSI": 4.28,
        "specific_energy": -47.760,
        "CHRatio": 0.4615,
        "CN": 45.21,    
    }
    group_map = {
        "Paraffin": ["N-Paraffin", "I-Paraffin"],
        "Naphthene": ["Naphthene"],
        "Aromatic": ["Aromatic"], 
        "Oxygenate": ["Oxygenate"]
    }
    group_limits = {
        "Paraffin": (0.001, 0.9),
        "Naphthene": (0.001, 0.9),
        "Aromatic": (0.001, 0.9),
        "Oxygenate": (0.0, 0.0),
    }
    
    # load experimental data for the distillation curve
    dist_curve = np.loadtxt(path_dist,delimiter=',',skiprows=1, dtype=float)

    # Open and load experimental data for the Phase Diagram
    with open(path_PD,'r') as f:
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
    
    #Add px_JetA and py_JetA to target_fuel_propberties
    target_fuel_properties.update({"px_JetA": px_JetA})
    target_fuel_properties.update({"py_JetA": py_JetA})

    target_fuel_properties.update({"dist_curve":dist_curve})
    return target_fuel_properties, group_map, group_limits

def validate_inputs(type_algorithm,filename,number_components,components,ratios,linear_props,target_fuel_properties, group_map, group_limits):
    models = ['NSGA_III']
    for group, (min_sum, max_sum) in group_limits.items():
        if (min_sum > max_sum):
            raise ValueError("minimaler Anteil ist größer als maximaler Anteil")
        
    if type_algorithm not in models:
        raise ValueError("Geben Sie entweder GA, PSO oder property an")

    if filename is None:
        raise ValueError("Path to input Datafile is missing!")
    
    if number_components is not None and not isinstance(number_components, int):
        raise TypeError("number_components must be an integer or None.")

    if components is not None and not isinstance(components, list):
        raise TypeError("components must be a list or None.")

    if ratios is not None and not isinstance(ratios, list):
        raise TypeError("ratios must be a list or None.")

    if target_fuel_properties is not None and not isinstance(target_fuel_properties, dict):
        raise TypeError("target_fuel_properties must be a dict or None.")

    if group_map is not None and not isinstance(group_map, dict):
        raise TypeError("group_map must be a dict or None.")

    if group_limits is not None and not isinstance(group_limits, dict):
        raise TypeError("group_limits must be a dict or None.")
    
    if type_algorithm == 'NSGA-III':
        if(number_components is None and components is None):
            raise ValueError("Either a number of components is needed or a list of fixed components")
        if(number_components is not None and components is not None):
            raise ValueError('Either a number of components or components must be None')
        elif(target_fuel_properties is None):
            raise ValueError("A dictionary of target fuel properties is needed")
        elif(group_map is None):
            raise ValueError("A group_map is needed")
        elif(group_limits is None):
            raise ValueError("Group_limits are needed")
        
    return True

def validate_fixed_components(fixed_components, df, group_map, group_limits):
    """
    Validiert fixe Components und prüft ob Gruppen-Constraints erfüllbar sind
    
    Args:
        fixed_components: Liste von fixen Componentsnamen oder None
        df: DataFrame mit Components
        group_map: Dict {group: [families]}
        group_limits: Dict {group: (min_sum, max_sum)}
    
    Returns:
        bool: True wenn valide, wirft ValueError bei Problemen
    """
    if fixed_components is None or len(fixed_components) == 0:
        return True
    
    # Prüfe ob alle Components existieren
    available_components = set(df['name'].tolist())
    for comp in fixed_components:
        if comp not in available_components:
            raise ValueError(f"Fixe Komponente '{comp}' existiert nicht in DataFrame")
    
    # Prüfe ob mindestens eine Komponente aus jeder ERFORDERLICHEN Gruppe dabei ist
    component_to_group = {}
    for idx, row in df.iterrows():
        comp_name = row['name']
        family = row['family']
        for group, families in group_map.items():
            if family in families:
                component_to_group[comp_name] = group
                break
    
    groups_covered = set()
    for comp in fixed_components:
        group = component_to_group.get(comp)
        if group:
            groups_covered.add(group)
    
    # Nur Required groups (min_sum > 0) müssen abgedeckt sein
    required_groups, not_required_groups = get_required_groups(group_limits)
    missing_required_groups = required_groups - groups_covered
    
    if missing_required_groups:
        raise ValueError(f"Fixe Components decken nicht alle erforderlichen Gruppen ab. "
                        f"Fehlende Gruppen (mit min_sum > 0): {missing_required_groups}")
    
    return True

def get_components_by_group(df, group_map, group_limits):
    """
    Designates a dictionary with every relevant group and the corresponding molecules

    Args:
        df: Dataframe
        group_map: dict of all hydrocarbon type names
        group_limits: dict with the limits for every type
    
    Returns:
        components_by_group: ordered list for all molecules in their respective group
    """
    
    components_by_group = {}
    for group, families in group_map.items():
        
        # Gruppen mit max_sum == 0 überspringen
        if group_limits.get(group, (0, 0))[1] <= 0:
            continue
        
        components = df[df['family'].isin(families)]['name'].tolist()
        components_by_group[group] = components
    
    return components_by_group

def precompute_group_indices(df, group_map):
    """
    Computes group-indizes for all components
    
    Args:
        df: DataFrame with Components
        group_map: Dict {group: [families]}
    
    Returns:
        component_to_group: Dict {component_name: group}
        name_to_idx: Dict {component_name: df_index}
    """
    component_to_group = {}
    name_to_idx = {name: idx for idx, name in enumerate(df['name'])}
    
    for idx, row in df.iterrows():
        comp_name = row['name']
        family = row['family']
        for group, families in group_map.items():
            if family in families:
                component_to_group[comp_name] = group
                break
    
    return component_to_group, name_to_idx

def get_required_groups(group_limits):
    """
    Returns the required groups, and those that are not required.

    Args:
        group_limits: Dict {group: (min_sum, max_sum)}
    
    Returns:
        required_groups: set of groups indicated by group_limits
        not_required_groups: the remaining groups
    """
    required_groups = set()
    not_required_groups = set()
    for group, (min_sum, max_sum) in group_limits.items():
        if min_sum > 1e-6:  # Gruppe ist erforderlich
            required_groups.add(group)
        if max_sum == 0: 
            not_required_groups.add(group)

    return required_groups, not_required_groups

def prepare_eval_data(components, df, linear_props, component_to_group, 
                     group_limits):
    """
    Prepares all data for evaluation
    
    Args:
        components: list of componentnames
        df: DataFrame with Components
        linear_props: list of properties
        component_to_group: Dict {component_name: group}
        group_limits: Dict {group: (min_sum, max_sum)}
    
    Returns:
        (sub_df, group_indices_list) or None on error
    """
    try:
        sub_df = df.set_index('name').loc[components].reset_index()
    except KeyError:
        return None
    
    if not all(prop in sub_df.columns for prop in linear_props):
        return None
    
    # Gruppen-Indizes für diese Componentskombination
    group_indices_list = []
    for group in group_limits.keys():
        indices = [i for i, comp in enumerate(components) 
                  if component_to_group.get(comp) == group]
        group_indices_list.append((indices, group_limits[group]))
    
    return sub_df, group_indices_list

def prepare_pd_data(components, df):
    """
    Prepares all data for evaluation 
    
    Args:
        components: list of componentnames
        df: DataFrame with Components
    
    Returns:
        sub_df or None on error
    """
    calc_prop = ['name','Tboil','Tcrit','Pcrit','Omega']
    try:
        sub_df = df.set_index('name').loc[components].reset_index()
    except KeyError:
        return None

    sub_df = sub_df[calc_prop]

    return sub_df

def init_ind(components_by_group, num_comp, group_limits, fixed_components=None):
    '''
    Initializes a single individual with added constraints

    Args:
        components_by_group: List of molecules ordered by group
        num_comp: number of components that are determined to be in the mixture
        group_limits:  Dict {group: (min_sum, max_sum)}
        fixed_components: array of predetermined components

    Returns:
        components: list of component names
        ratios: corresponding designated fraction
    '''
    #Modus 1: components are given, fractions are varied
    if fixed_components is not None and len(fixed_components) > 0 : 
        components = fixed_components.copy()
        if len(components)> num_comp:
            raise ValueError(f"Anzahl fixer Components ({len(components)}) "
                           f"größer als max_blend_size ({num_comp})")
        while True:
            #create vector with length num_comp and sum of 1
            ratios = np.random.dirichlet(np.ones(num_comp)) 
            if np.all(ratios >= 0.001):
                break
    else:   #Modus 2: everything is variable, only relevant groups are considered
        components = []
        required_groups, _ = get_required_groups(group_limits)
        for group in required_groups:
            val = random.randint(1,len(components_by_group[group])-1) #select random group
            new_comp = components_by_group[group][val]
            while new_comp in components:   #select random component from group, as long as not already present in mixture
                val = random.randint(1,len(components_by_group[group])-1)
                new_comp = components_by_group[group][val]
            components.append(new_comp)
            print(components_by_group[group][val])
        
        num_remaining_comp = num_comp - len(components)
        while num_remaining_comp > 0:
            group = random.choice(list(components_by_group.keys()))   #select random group
            val = random.randint(1,len(components_by_group[group])-1) 
            new_comp = components_by_group[group][val]
            while new_comp in components: #select random component from group, as long as not already present in mixture
                val = random.randint(1,len(components_by_group[group])-1)
                new_comp = components_by_group[group][val]

            components.append(new_comp)
            num_remaining_comp = num_comp - len(components) #calculate remaining slots
            print(components_by_group[group][val])
        while True:
            #create vector with length num_comp and sum of 1
            ratios = np.random.dirichlet(np.ones(num_comp)) 
            if np.all(ratios >= 0.001):
                break        
    print("--------")
    components = list(dict.fromkeys(components))[:num_comp]
    
    return components, ratios

def pop_init(components_by_group,num_comp,group_limits,fixed_components,n=100):
    '''
    For fixed_components != None, a spacing operation is conducted
    ensuring equal spacing betwen individuals and covering the solutionspace
    within the fixed limits

    Args:
        components_by_group: List of molecules ordered by group
        num_comp: number of components that are determined to be in the mixture
        group_limits:  Dict {group: (min_sum, max_sum)}
        fixed_components: array of predetermined components
        n: Population size

    Returns:
        pre_calc: Population with components and corresponding ratios
    '''
    POPSIZE = n
    limits = []
    for _ in range(num_comp):
        #add the maximum limits for each components
        limits.append((0,1))
    
    iterations = 100
    civilization = []
    comps = []
    for n in range(POPSIZE):
        components,ratios = init_ind(components_by_group, num_comp, group_limits, fixed_components)
        comps.append(components)
        civilization.append(ratios)

    # if fixed_components is not None:
    if fixed_components == num_comp:
        population = np.asarray(civilization)
        #using "k-nearest neighbours" to find the individuals, that will influence the central individual the most
        neigh = NearestNeighbors(n_neighbors=num_comp+1) 
        neigh.fit(population)
        dt = 0.1
        new_pop = population
        for iter in range(1,iterations):
            diff = []
            population = new_pop
            for indi in population:
                #return the k-nearest for each individual in the population
                neigh.fit(population)
                dist = neigh.kneighbors([indi],return_distance=False)
                diff.append(dist)

            for i in range(len(diff)):
                # For every individual the momentary force is calculated
                l = len(diff[i][:][0])-1
                center = population[diff[i][0][0]]

                p_sum = np.zeros(num_comp)
                vec = np.zeros(num_comp)
                for item in population[diff[i][0][1:]]:
                    #calculate the vector between neighbour and center individual
                    vec = (center-item)
                    vec = np.around(vec,8)

                    dist = np.sqrt((center[0]-item[0])**2+(center[1]-item[1])**2+(center[2]-item[2])**2) #distance
                    a = vec / (dist ** 2) *10 #tuned acceleration v/r^2 *tuningparameter
                    v = a * dt/2.0
                    dp = v * dt     #delta displacement happening during this step
                    p_sum += dp

                # testing whether the individum still lies within the valid region
                prelim = population[i] + p_sum
                test = [limits[i][0]>prelim[i] or limits[i][1]<prelim[i] for i in range(len(center))]

                if any(test):
                    new_pop[i] = population[i] - p_sum*0.0001
                else:
                    new_pop[i] = prelim
    else: 
        population = np.asarray(civilization)

    ratios = population
    pre_calc = [comps,ratios.tolist()]
    
    return pre_calc
