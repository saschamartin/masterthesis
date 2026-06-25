import sys
import numpy as np
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils.properties import calculate_mixture_properties
from utils.setup import (prepare_eval_data,prepare_pd_data,get_required_groups)
sys.path.append("C:/Users/Skulleton/Documents/#Uni/#Masterthesis/Code/phase/DistillationCurve/distillation")
from distillation import run as run_dist
from two_phase_lib import run as run_PD

# ----------- Fitness Operators ---------------
def fitness_function(individual, df, linear_props, b_target, eData_dist, eData_PD, group_map, 
                    group_limits, MIN_BOUND, MAX_BOUND, PENALTY_SCALE,
                    component_to_group=None, fixed_components=None):
    """ 
    Args:
        individual: [components, ratios] - List with Componentnames and Fractions
        df: DataFrame with Components
        linear_props: List of Properties
        b_target: Array of reference Values
        group_map: Dict {group: [families]}
        group_limits: Dict {group: (min_sum, max_sum)}
        MIN_BOUND: MIN Fraction
        MAX_BOUND: MAX Fraction
        PENALTY_SCALE: Sclae for Penalties
        component_to_group: Optionally a precomputed Mapping (Performance)
        fixed_components: Liste of fixed Components
    
    Returns:
        fitness: Float (lower is better)
    """

    components, ratios = individual
    # print("Number Components: ",len(components),"; Ratios: ",len(ratios))
    ratios = np.array(ratios, dtype=float)
    error = np.ones((len(components),1))
    
    # Validation: with set fixed_components these have to be included
    if fixed_components is not None:
        if not all(fc in components for fc in fixed_components):
            return error*1e10
    
    # 1. Validation of Components
    try:
        sub_df_blend = df.set_index('name').loc[components].reset_index()
    except KeyError:
        return error*1e10
    
    if not all(prop in sub_df_blend.columns for prop in linear_props):
        return error*1e10
    
    # preparing data
    eval_data = prepare_eval_data(components, df, linear_props, 
                                  component_to_group, group_limits)
    if eval_data is None:
        return error*1e10
    
    sub_df_pd = prepare_pd_data(components, df)
    
    sub_df_blend, group_indices_list = eval_data
    # 2. Normalization of mixture
    s = np.sum(ratios)
    # if s <= 0:
    #     return error*1e10
    x_norm = ratios / s
    
    # 3.1 Calculate Mixtureproperties
    predicted = calculate_mixture_properties(sub_df_blend, x_norm, linear_props)
    
    # 3.2 Calculate Distillationcurve
    # 3.2.1 Check for valid mixture range (!) x_i>0
    validity = [item>0.0 for item in ratios]
    if all(validity):
        px_dist = calculate_dist(sub_df_pd,ratios)
        px_PD,py_PD = calculate_PD(sub_df_pd,ratios)
        rel_errors_dist = area(px_dist,eData_dist)
        # rel_frech = frechet_distance(px_PD,py_PD)
        rel_errors_PD = distance_PD(px_PD,py_PD,eData_PD)
    else:
        return error*1e10
        

    # 4. Calculate relative errors
    safe_b = np.where(b_target == 0, 1.0, b_target)
    rel_errors = np.abs((predicted - b_target) / safe_b)
    rel_errors = np.append(rel_errors,rel_errors_dist)
    rel_errors = np.append(rel_errors,rel_errors_PD)

    # max_error = np.max(rel_errors)
    # avg_error = np.mean(rel_errors)
    
    # # 5. Constraint-Penalties
    penalty = check_constraints_optimized(x_norm, group_indices_list, group_limits,
                                          MIN_BOUND, MAX_BOUND, PENALTY_SCALE)
    
    '''
    Error functions
    '''
    f = lambda t: t**2
    core_fitness = [f(err) for err in rel_errors]
    fitness = [c + penalty for c in core_fitness]
    return fitness

def calculate_PD(df,ratios):
    '''
    Calculation of the phasediagram 

    Args:
        df: Dataframe of the molecules in the mixture
        ratios: fractions in the mixture
    Returns:
        px: bubble point curve 
        py: dew point curve
    '''
    phys_file = "./Data/physDatabank.csv"
    
    result = run_PD(
        df,
        ratios,
        phys_file,
        verbose=False
    )
    
    px = result.px_solution[:,:2]
    py = result.py_solution[:,:2]
    if px[0][0] != 0:
        px = np.trim_zeros(px,trim="b")
    if py[0][0] != 0:
        py = np.trim_zeros(py,trim="b")

    return px,py

def calculate_dist(df,ratios):
    '''
    Calculation of the distillation curve

    Args:
        df: Dataframe of the molecules in the mixture
        ratios: fractions in the mixture

    Returns:
        px: coordinates of curve
    '''
    phys_file = "./Data/physDatabank.csv"
    result = run_dist(
            df,
            ratios,
            phys_file,
            verbose=False
        )
    
    px = result.px_solution[:,[5,1]]
    px = px[:-1,:]
    
    with open("result_px.txt","w") as f:
        for row in px:
            f.write(' '.join([str(a) for a in row]) + '\n')
    return px

def area(px,expData):
    '''
    Determination of the area between the reference curve and the experimental Data

    Args:
        px: calculated distillation curve
        expData: experimental distillation Curve
    
    Return:
        normalized area
    '''
    # ------------------------
    # normalization of curves
    # ------------------------
    expData[:,1] = expData[:,1] + 273.15

    # ------------------------
    #   area between curves
    # ------------------------
    c = np.trapezoid(px[:,1],px[:,0])
    e = np.trapezoid(expData[:,1],expData[:,0])
    d = abs(c-e)/e
    return np.abs(d)

def distance_PD(px,py,eData):
    """
    Calculate the deviation between the Phase Diagram curves
    
    Args:
        px: Array for bubble point curve (P,T,zL,zV,x(1),...,x(end))
        px: Array for dew point curve (P,T,zL,zV,y(1),...,y(end))
        eData: Experimental data of the Phase Diagram ("JetA-Phasediagram.csv)

    Return:
        root mean squarred error
    """
    exp_px_norm = np.zeros((len(eData[0]),2))
    exp_py_norm = np.zeros((len(eData[1]),2))
    px_norm = np.zeros((len(px),2))
    py_norm = np.zeros((len(py),2))
    exp_px = eData[0]
    exp_py = eData[1]

    T_max = max(max(exp_px[:,1]),max(exp_py[:,1]),max(px[:,1]),max(py[:,1]))
    p_max = max(max(exp_px[:,0]),max(exp_py[:,0]),max(px[:,0]),max(py[:,0]))

    exp_px_norm[:,0] = exp_px[:,0]/p_max
    exp_px_norm[:,1] = exp_px[:,1]/T_max
    exp_py_norm[:,0] = exp_py[:,0]/p_max
    exp_py_norm[:,1] = exp_py[:,1]/T_max

    px_norm[:,0] = px[:,0]/p_max
    px_norm[:,1] = px[:,1]/T_max
    py_norm[:,0] = py[:,0]/p_max
    py_norm[:,1] = py[:,1]/T_max


    sum = 0.0
    n = 0
    for e in exp_px_norm:
        min_dist = float('inf')
        nearest = []
        for c in px_norm:
            # if c[0]>=exp_px_norm[0,0] and c[0]<=exp_px_norm[-1,0]: 
            dist = np.sqrt((e[0]-c[0])**2+(e[1]-c[1])**2)
            if dist < min_dist:
                nearest = c
                min_dist = dist
        sum += np.sqrt((e[0]-nearest[0])**2+(e[1]-nearest[1])**2)
        n+=1
    rms_px = np.sqrt(sum/n)
    sum = 0.0
    n = 0
    for e in exp_py_norm:
        min_dist = float('inf')
        nearest = []
        for c in py_norm:
            # if c[0]>=exp_py_norm[0,0] and c[0]<=exp_py_norm[-1,0]: 
            dist = np.sqrt((e[0]-c[0])**2+(e[1]-c[1])**2)
            if dist < min_dist:
                nearest = c
                min_dist = dist
        sum += np.sqrt((e[0]-nearest[0])**2+(e[1]-nearest[1])**2)
        n+=1
    rms_py = np.sqrt(sum/n)
    rms = rms_px+rms_py #normalized for number of calculation points. Number can vary -> inaccurate solution

    return rms

def check_constraints_optimized(x_ratios, group_indices_list, group_limits, 
                               MIN_BOUND, MAX_BOUND, PENALTY_SCALE):
    """
    Constraint check with precomputed Indizes
    
    Args:
        x_ratios: Array of normalized ratios
        group_indices_list: list of (indices, (min, max)) tuples
        group_limits: Dict {group: (min_sum, max_sum)}
        MIN_BOUND: MIN Fraction
        MAX_BOUND: MAX Fraction
        PENALTY_SCALE: Scaling Factor
    
    Returns:
        penalty: Float
    """
    penalty = 0.0
    
    # Group-Constraints (precomputed)
    for _,(indices, (min_sum, max_sum)) in zip(group_limits.keys(), group_indices_list):
        if not indices:
            continue
        group_sum = np.sum(x_ratios[indices])
        if group_sum < min_sum:
            penalty += PENALTY_SCALE * (min_sum - group_sum) ** 2
        elif group_sum > max_sum:
            penalty += PENALTY_SCALE * (group_sum - max_sum) ** 2
    
    return penalty

def selectBest(individuals,k):
    '''
    Adapted selection algorithm from the deap-module
    fitness = indi.fitness.values given as vector -> best individum has the lowes summed up score

    Args:
        individuals: population creator class
        k: number of returned individuals

    Returns: 
        sorted list of k individuals best to worst
    '''
    def fitness_sum(i):
        sum = 0
        for f in i.fitness.values:
            sum += f
        return sum
    
    return sorted(individuals, key=fitness_sum,reverse=False)[:k]


# ------------ Genetic Operators ----------------------
def crossover(ind1, ind2, progress, MIN_BOUND, MAX_BOUND, eta_end=300):
    """
    Optimized SBX crossover with failsafes
    Args:
        ind1: first individual
        ind2: second individual
        progress: ratio of progress to max_gen
        MIN_BOUND: MIN Fraction
        MAX_BOUND: MAX Fraction
        eta_end: final value for continually adapted eta
    Returns:
        ind1,ind2: offspring individuals
    """

    """monotone increase of eta (spread operator)"""
    eta_c = 30
    # eta_c = 110.77
    # eta_c_init = 2
    # eta_c = 2+progress*(eta_end-eta_c_init)
    

    ratios1 = np.array(ind1[1], dtype=float)
    ratios2 = np.array(ind2[1], dtype=float)
    
    u = np.random.random(len(ratios1))
    beta = np.where(u <= 0.5,
                (2 * u) ** (1.0 / (eta_c + 1)),
                (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1)))
    
    c1 = 0.5 * ((1 + beta) * ratios1 + (1 - beta) * ratios2)
    c2 = 0.5 * ((1 - beta) * ratios1 + (1 + beta) * ratios2)

    # Ensuring the limits are adehered to
    c1 = [min(max(c,MIN_BOUND),MAX_BOUND) for c in c1]
    c2 = [min(max(c,MIN_BOUND),MAX_BOUND) for c in c2]

    ind1[1] = (c1 / np.sum(c1)).tolist()
    ind2[1] = (c2 / np.sum(c2)).tolist()
    
    # Components-Crossover
    # safety feature that no component can exist two times in the same mixture
    size = min(len(ind1[0]), len(ind2[0]))
    if size > 1 and random.random() < 0.5:
        cxpoint = random.randint(1, size - 1)
        if not (any(e in ind2[0][:cxpoint] for e in ind1[0][cxpoint:]) or any(e in ind1[0][:cxpoint] for e in ind2[0][cxpoint:])):
            ind1[0][cxpoint:], ind2[0][cxpoint:] = ind2[0][cxpoint:], ind1[0][cxpoint:]
    
    return ind1, ind2

def mutate(individual, generation, components_by_group, 
            component_to_group, MIN_BOUND, MAX_BOUND, NGEN, group_limits,
            fixed_components):
    """
    Adaptive Mutation
    
    Args:
        individual: DEAP Individual
        generation: Current Generation
        components_by_group: Dict {group: [components]}
        component_to_group: Dict {component: group}
        MIN_BOUND: Min Fraction
        MAX_BOUND: Max Fraction
        NGEN: Total Number of Generations
        fixed_components: List of fixed components
    Returns:
        individual: creator object
    """
    components, ratios = individual
    
    eta_m = 20
    # eta_m = 0.15
    # eta_m = 300*(1-(generation)/NGEN)
    
    # Componentsmutation only when no fixed components
    if fixed_components is None or len(fixed_components) == 0:
        if random.random() < 0.2:
            required_groups, not_required_groups = get_required_groups(group_limits)
            
            # Count Components à group
            current_groups = {}
            for i, comp in enumerate(components):
                group = component_to_group.get(comp)
                if group:
                    if group not in current_groups:
                        current_groups[group] = []
                    current_groups[group].append(i)
            
            replaceable_indices = []
            for group, indices in current_groups.items():
                if group in required_groups:
                    # Necesary group only when > 1
                    if len(indices) > 1:
                        replaceable_indices.extend(indices)
                else:
                    # Optional Group: all components are replacable
                    replaceable_indices.extend(indices)
            
            if not replaceable_indices:
                # Fallback: all components are replacable
                replaceable_indices = list(range(len(components)))
            
            # Chose random component for exchange
            idx = random.choice(replaceable_indices)
            all_components = [comp for comps in components_by_group.values() 
                             for comp in comps]
            available = list(set(all_components) - set(components))
            
            if available:
                components[idx] = random.choice(available)
    
    # Fractionmutation
    ratios = np.array(ratios) + np.random.normal(0, eta_m, len(ratios))
    ratios = np.clip(ratios, MIN_BOUND, MAX_BOUND)
    ratios /= np.sum(ratios)
    
    individual[0], individual[1] = components, ratios.tolist()
    return individual,

def adaptive_parameters(gen, NGEN, dim):
    """
    Adaption of Genetic Parameters
    Args:
        gen: current generation
        NGEN: maximum generation
        dim: dimension of objective space
    Returns:
        cxpb: crossover probability
        mutpb: mutation probability
        progress: 
    """
    # p = dim/50
    # pn = p - (gen-1)* p/(NGEN-1)

    progress = gen / NGEN
    # mutpb = pn/dim #0.5 * (1 - progress * 0.7)  # 0.5 -> 0.15
    # cxpb = 0.6 + progress * 0.4 # 0.6 -> 1
    # mutpb = 0.19
    # cxpb = 0.93
    mutpb = 1/dim
    cxpb = 1
    return cxpb, mutpb, progress
    
