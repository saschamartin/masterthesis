import sys
sys.path.insert(0, "./NSGA/utils/output")
import time, random, multiprocessing
import numpy as np
from functools import partial
from deap import base, tools, creator
from utensils.output import (export_figure_dist,
                          export_figure_PD,
                          print_final_results,
                          plot_convergence,)
# from populating import pop_init
from utensils.utils import (fitness_function,
                        crossover,
                        mutate,
                        adaptive_parameters,
                        selectBest)
from utensils.setup import (get_components_by_group,
                        precompute_group_indices,
                        get_required_groups,
                        pop_init,
                        validate_fixed_components,)


def init_population(components_by_group,num_comp,group_limits,fixed_components,POPSIZE):
    '''
        Initializing population
    '''
    pre_calc = pop_init(components_by_group,num_comp,group_limits,fixed_components,POPSIZE)
    # creating individuals from the calculated population
    population = [creator.Individual([pre_calc[0][i],pre_calc[1][i]]) for i in range(len(pre_calc[0]))]

    return population


def eval_ind_nsga(individual, df, linear_props, b_target, eData_dist, eData_PD, 
                      group_map, group_limits,
                      MIN_BOUND, MAX_BOUND, PENALTY_SCALE,
                      component_to_group, fixed_components):
    """
    GA-Wrapper for the fitness function
    """
    fitness = fitness_function(
        individual, df, linear_props, b_target, eData_dist, eData_PD, group_map, 
        group_limits, MIN_BOUND, MAX_BOUND, PENALTY_SCALE,
        component_to_group, fixed_components
    )
    return fitness

def NSGA_algo(max_blend_size, df, target_fuel_properties, group_map, group_limits,
                POP_SIZE=800, NGEN=800, ELITE_SIZE=1,
                PENALTY_SCALE=1e6, MIN_BOUND=0.001, MAX_BOUND=1.0, 
                use_all_cpus=True, early_stop_gen=100, fixed_components=None):
    export = True
    dist = "dist_curve"
    pd_px = "px_JetA"
    pd_py = "py_JetA"
    eData_dist = target_fuel_properties[dist]
    eData_PD = [target_fuel_properties[pd_px],target_fuel_properties[pd_py]]

    best_scores = []
    NOBJ = len(target_fuel_properties)-1

    start_time = time.time()

    #creating Optimization problem with minimization
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights= (-1.0,)*NOBJ)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness = creator.FitnessMin)

    # Precalculations
    print(" Preparing data")
    components_by_group = get_components_by_group(df, group_map, group_limits)
    component_to_group, name_to_idx = precompute_group_indices(df, group_map)
    required_groups, not_required_groups = get_required_groups(group_limits)
    total_groups = len(required_groups)
    # Validations
    if fixed_components is not None and len(fixed_components) > 0:
        validate_fixed_components(fixed_components, df, group_map, group_limits)
        NUM_COMPONENTS_TO_SELECT = len(fixed_components)
        print(f"\n Modus: Fixe Components ({NUM_COMPONENTS_TO_SELECT})")
        print(f"   Components: {', '.join(fixed_components)}")
    else:
        NUM_COMPONENTS_TO_SELECT = max_blend_size
        required_groups, not_required_groups = get_required_groups(group_limits)
        if NUM_COMPONENTS_TO_SELECT < len(required_groups):
            raise ValueError(f"max_blend_size ({NUM_COMPONENTS_TO_SELECT}) muss mindestens "
                           f"so groß sein wie die Anzahl der erforderlichen Gruppen ({len(required_groups)})")
        print(f"\n Modus: Variable components (max {NUM_COMPONENTS_TO_SELECT})")
        print(f"   Required groups (min_sum > 0): {required_groups if required_groups else 'keine'}")
        
        optional_groups = set(group_map.keys()) - required_groups- not_required_groups
        print(f"   Optional Groups (min_sum = 0): {optional_groups if optional_groups else 'keine'}")
    
    print(f"\n Available components per group")
    for group, comps in components_by_group.items():
        print(f"   {group}: {len(comps)} Components")
    
    #initialize blend target properties
    [target_fuel_properties.pop(key) for key in ["dist_curve","px_JetA","py_JetA"]]
    linear_props = list(target_fuel_properties.keys())
    b_target = np.array([target_fuel_properties[prop] for prop in linear_props], 
                       dtype=float)


# Toolbox Setup
    ''' For Original NSGA-III'''
    layer1_ref = tools.uniform_reference_points(NOBJ,p=1,scaling=1)
    layer2_ref = tools.uniform_reference_points(NOBJ,p=1,scaling=0.75)
    layer3_ref = tools.uniform_reference_points(NOBJ,p=1,scaling=0.5)
    ref_points = np.concatenate([layer1_ref,layer2_ref,layer3_ref])

    ref_points = tools.uniform_reference_points(NOBJ,p=2,scaling=1)
    toolbox = base.Toolbox()

    toolbox.register("individual", tools.initRepeat, creator.Individual)

    evaluate_partial = partial(
        eval_ind_nsga,
        df=df,
        linear_props=linear_props,
        b_target=b_target,
        eData_dist=eData_dist,
        eData_PD=eData_PD,
        group_map=group_map,
        group_limits=group_limits,
        MIN_BOUND=MIN_BOUND,
        MAX_BOUND=MAX_BOUND,
        PENALTY_SCALE=PENALTY_SCALE,
        component_to_group=component_to_group,
        fixed_components=fixed_components
    )

    toolbox.register("evaluate",evaluate_partial)
    toolbox.register("mate", crossover)

    mutate_partial = partial(mutate,
                             components_by_group=components_by_group, 
                             component_to_group=component_to_group,
                             MIN_BOUND=MIN_BOUND,
                             MAX_BOUND=MAX_BOUND,
                             NGEN=NGEN,
                             group_limits=group_limits,
                             fixed_components=fixed_components) #wraper for the mutate function

    toolbox.register("mutate", mutate_partial)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points,nd="standard")

    num_processes = multiprocessing.cpu_count() if use_all_cpus else 1
    pool = multiprocessing.Pool(processes=num_processes)
    toolbox.register("map", pool.map)


    best_ratios = []
    best_fitness = []
    try:
        # Initialpopulation
        print(f"\n🚀 Create initial population ({POP_SIZE})...")
        pop = init_population(components_by_group,NUM_COMPONENTS_TO_SELECT,group_limits,fixed_components,POP_SIZE)
        # first evaluation to initialize population
        fits = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit
        
        best_fitness_history = []
        stagnation_counter = 0
        last_best_fitness = np.ones(NOBJ)*float('inf')
        epsilon = 1e-3
        
        print(f"\n🎯 GA läuft auf {num_processes} processors")
        print(f"{'Gen':<6} {'Best':<12} {'Avg':<12} {'Stagnation':<12} {'Zeit/Gen'}")
        print("-" * 70)
        
        gen_start = time.time()
        iteration = 1
        for gen in range(NGEN):
            print("Generation: ", iteration)
            # Adaptive Parameters, probability of crossover and mutation
            cxpb, mutpb, progress = adaptive_parameters(gen, NGEN, NOBJ)
            
            offspring = [] 
            intermediate = list(map(toolbox.clone,pop)) #inhibit any misshandling of fitnessvalues from original population
            # create a second vector with only the individuals for crossover
            # preserve original population vector
            
            elites = list(map(toolbox.clone,selectBest(pop,ELITE_SIZE)))
            for i in range(POP_SIZE):
                if random.random() < cxpb:
                    offspring.append(intermediate[i])
            #crossover
            for i in range(1,len(offspring),2):
                crossover(offspring[i-1], offspring[i],progress,MIN_BOUND,MAX_BOUND)
                #delete fitness values of individuals -> reevaluation is done
                del offspring[i-1].fitness.values
                del offspring[i].fitness.values
            # add previous population back
            [offspring.append(ind) for ind in pop]

            elite_index = sorted([offspring.index(e) for e in elites],reverse=True)
            [offspring.pop(i) for i in elite_index]
            # Mutation
            for ind in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(ind, gen)
                    del ind.fitness.values
            # [offspring.append(e) for e in elites]

            
            # Evaluation of all individuals without valid fitness values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                fits = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fits):
                    ind.fitness.values = fit
            
            #Select Operator
            pop = toolbox.select(offspring, POP_SIZE-ELITE_SIZE)
            [pop.append(e) for e in elites]
            # Statistik
            best = selectBest(pop, 1)[0]
            best_fitness = best.fitness.values[:]
            best_fitness_history.append(best_fitness)
            avg_fitness = np.mean([ind.fitness.values[0] for ind in pop])
            
            if export:
                # Export the best individual (Figure)
                export_figure_dist(best,df,eData_dist,gen)
                export_figure_PD(best,df,eData_PD,gen)
                # Export the best individual (Individual)
                best_ratios.append(best)

            # Stagnation Detection
            if abs(sum(best_fitness) - sum(last_best_fitness)) < epsilon:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            last_best_fitness = best_fitness
            best_scores.append(best_fitness)
            gen_time = time.time() - gen_start
            
            if gen % 50 == 0 or gen < 10:
                print(f"{gen:<6} {sum(best_fitness):<12.6f} {avg_fitness:<12.6f} "
                      f"{stagnation_counter:<12} {gen_time:.2f}s")
            
            gen_start = time.time()
            iteration+=1
            if stagnation_counter >= early_stop_gen:
                print(f"\n⚠️  Early stop due to stagnationo at Generation {gen}")
                break
            convergence_check = [bf<epsilon for bf in best_fitness]
            if all(convergence_check):
                print(f"\n Early stop due to found Optimum")
                break


        best = selectBest(pop, 1)[0]
        components, ratios = best
    
        if export:
            with open("Export/individuals.txt","w") as f:
                for row in best_ratios:
                    n = len(components)
                    for i in range(n):
                        f.write(f"{row[0][i]},{row[1][i]}")
                        if i < n-1:
                            f.write(",")
                    f.write("\n")
            with open("Export/fit_convergence.txt","w") as o:
                o.write(f"{"density"},{"molar_mass"},{"viscosity"},{"TSI"},{"spec.E"},{"CHRatio"},{"CN"},{"DistillationCurve"},{"PhaseDiagram"}\n")
                # o.write(f"{"density"},{"molar_mass"},{"viscosity"},{"spec.E"},{"CHRatio"},{"CN"},{"DistillationCurve"},{"PhaseDiagram"}\n")
                for row in best_scores:
                    for item in row:
                        o.write(f"{item}")
                        if item != row[-1]:
                            o.write(f",")
                    o.write("\n")
        
        # Ausgabe (gemeinsame Funktion)
        elapsed_time = time.time() - start_time
        max_err, mean_err = print_final_results(
            components, ratios, df, linear_props, b_target, eData_dist,
            group_map, component_to_group, elapsed_time, "GA"
        )
        
        plot_convergence(best_scores)   
        return [components, ratios], max_err, mean_err
        
    finally:
        pool.close()
        pool.join()