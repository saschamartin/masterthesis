import numpy as np
from numpy import random
from deap import creator, tools
from sklearn.neighbors import NearestNeighbors
from utils import (get_required_groups,)

def init_ind(components_by_group, num_comp, group_limits, fixed_components=None):
    #Modus 1: es sind die Komponenten Vorgegeben und nur die Mengenverhältnisse werden angepasst
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
    else:   #Modus 2: alles ist variabel, aber es werden nur relevante Gruppen betrachtet 
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
            num_remaining_comp = num_comp - len(components) #berechne die übrigbleibenden slots
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

    if fixed_components is not None:
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

