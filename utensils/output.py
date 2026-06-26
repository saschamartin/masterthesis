from utensils.setup import (prepare_pd_data)
from utensils.utils import (calculate_dist,
                         calculate_PD)
from utensils.properties import (calculate_mixture_properties)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

def export_figure_dist(best,df,eData,gen):
    '''
    Exports the distillation curve by calculating it from the given best individual

    Args:
        best: creator object of the best individual of the population
        df: dataframe with the physical molecule properties
        eData: experimental distillation curve data
        gen: current generation
    Returns:
        None
    '''
    path_export = "Export"

    components = best[0]
    ratios = best[1]

    sub_df = prepare_pd_data(components, df)
    px = calculate_dist(sub_df,ratios)

    title = f"Distillation Curve - Generation: {gen+1}\n"
    local = f""
    for i in range(len(components)):
        local += f"{components[i]}: {np.round(ratios[i]*100,2)}%"

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(eData[:,0],eData[:,1]+273.15,"o", label="Jet-A Experimental")
    ax.plot(px[:,0],px[:,1],label="Calculation")
    ax.set_ylim(380,580)
    ax.set_xlabel("Volumefraction [%]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(title)
    plt.legend(loc="upper left")
    # plt.show()
    fig.savefig(f"{path_export}/{str(gen+1)}_generation_dist.png")
    plt.close()

def export_figure_PD(best,df,eData,gen):
    '''
    Exports the phase diagram by calculating it from the given best individual

    Args:
        best: creator object of the best individual of the population
        df: dataframe with the physical molecule properties
        eData: experimental distillation curve data
        gen: current generation
    Returns:
        None
    '''
    exp_px = eData[0]
    exp_py = eData[1]
    path_export = "Export"

    components = best[0]
    ratios = best[1]

    sub_df = prepare_pd_data(components, df)
    px,py = calculate_PD(sub_df,ratios)
    title = f"Phase Diagram - Generation: {gen+1}\n"
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(exp_px[:,1],exp_px[:,0],"ro", label="Jet-A Experimental-bubble")
    ax.plot(exp_py[:,1],exp_py[:,0],"bo", label="Jet-A Experimental-dew")
    ax.plot(px[:,1],px[:,0],color="orange",label="Bubble point curve")
    ax.plot(py[:,1],py[:,0],color="cyan",label="Dew point curve")
    ax.set_ylim(0,40)
    ax.set_xlim(0,900)
    ax.set_ylabel("Pressure [atm]")
    ax.set_xlabel("Temperature [K]")
    ax.set_title(title)
    plt.legend(loc="upper left")
    fig.savefig(f"{path_export}/{str(gen+1)}_generation_PD.png")
    plt.close()

def print_final_results(components, final_ratios, df, linear_props, b_target, eData,
                       group_map, component_to_group, elapsed_time, 
                       algorithm_name=""):
    """
    Ouput function for final result
    
    Args:
        components: List of component names
        final_ratios: Array of final fractions
        df: DataFrame with components
        linear_props: List with properties
        b_target: Array of reference values
        group_map: Dict {group: [families]}
        component_to_group: Dict {component_name: group}
        elapsed_time: Running time in seconds
        algorithm_name: Name des Algorithmus
    
    Returns:
        max error
        mean error
    """
    sub_df = df.set_index('name').loc[components].reset_index()
    predicted_final = calculate_mixture_properties(sub_df, final_ratios, linear_props)
    safe_b = np.where(b_target == 0, 1.0, b_target)
    rel_errors_pct = np.abs((predicted_final - b_target) / safe_b) * 100
    
    print("\n" + "=" * 80)
    print(f"📊 FINAL RESULT {f'({algorithm_name})' if algorithm_name else ''}")
    print("=" * 80)
    print(f"⏱️  Duration: {elapsed_time:.1f} s")
    print(f"📉 Max. Deviation: {np.max(rel_errors_pct):.2f}%")
    print(f"📊 Mean. Deviation: {np.mean(rel_errors_pct):.2f}%\n")
    
    # Grouping Components
    grouped_components = {}
    for comp, ratio in zip(components, final_ratios):
        group = component_to_group.get(comp, "Unknown")
        if group not in grouped_components:
            grouped_components[group] = []
        family = df.loc[df['name'] == comp, 'family'].values[0]
        grouped_components[group].append((comp, family, ratio))
    
    print(f"{'Components':<30} {'Family':<20} {'Ratio (%)'}")
    print("-" * 70)
    
    for group in sorted(grouped_components.keys()):
        print(f"\n{group}:")
        group_total = 0
        for comp, family, ratio in grouped_components[group]:
            print(f"  {comp:<28} {family:<20} {ratio*100:>8.2f}%")
            group_total += ratio
        print(f"  {'Sum ' + group:<28} {'':<20} {group_total*100:>8.2f}%")
    
    print("\nProperties:")
    print("-" * 70)
    for prop, target, pred, err_pct in zip(linear_props, b_target, 
                                           predicted_final, rel_errors_pct):
        status = "✓" if err_pct <= 10 else "✗"
        print(f"{prop:<25} Target: {target:>10.4f}  Actual state: {pred:>10.4f}  "
              f"Δ={err_pct:>6.2f}%  {status}")
    
    print("=" * 80)
    
    return np.max(rel_errors_pct), np.mean(rel_errors_pct)

def plot_convergence(best_scores):
    plt.plot(best_scores)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("Convergence")
    plt.savefig("convergence.png")

def plotDist(px,eData):
    CtoK = 273.15 #Centigrade to Kelvin
    plt.figure(figsize=(8,6))
    plt.plot(px[:,0],px[:,1], label='Distillation curve')
    plt.plot(eData[:,0],eData[:,1]+CtoK,"o",label='Jet-A Reference')

    plt.xlabel("Volume [%]")
    plt.ylabel("Temperature [K]")

    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig("DistillationCurve.png")
    return plt


