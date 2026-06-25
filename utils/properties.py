import numpy as np
import re

def calculate_nonlinear_cetane_number(sub_df, ratios):
    """
    Berechnet CN_mix = sum(r_i * beta_i * CN_i) / sum(r_i * beta_i)
    """
    CN = sub_df["CN"].to_numpy(dtype=float)
    beta = (
        sub_df["beta_i"].to_numpy(dtype=float)
        if "beta_i" in sub_df.columns
        else np.ones_like(CN)
    )

    numerator = np.sum(ratios * beta * CN)
    denominator = np.sum(ratios * beta)

    return numerator / denominator if denominator > 0 else np.nan

def parse_formula(formula):
    """
    Extrahiert die Anzahl der C- und H-Atome aus der chemischen Formel
    Beispiele: cyC9H18, i-C10H22, n-C10H22, C8H18
    """
    # Suche nach C gefolgt von einer Zahl
    c_match = re.search(r'C(\d+)', formula)
    c_atoms = int(c_match.group(1)) if c_match else 0
    
    # Suche nach H gefolgt von einer Zahl
    h_match = re.search(r'H(\d+)', formula)
    h_atoms = int(h_match.group(1)) if h_match else 0
    
    return c_atoms, h_atoms

def calculate_nonlinear_ch_ratio(sub_df, ratios):
    """
    Berechnet das gewichtete H/C-Verhältnis einer Mischung aus einem DataFrame.
    
    Parameters:
    - sub_df: pandas DataFrame mit mindestens einer Spalte 'formula'
    - ratios: numpy array oder Liste der Gewichte (z.B. Massenanteile, Summe muss nicht 1 sein)
    
    Returns:
    - gewichtetes H/C-Verhältnis der Mischung
    """
    if len(sub_df) != len(ratios):
        raise ValueError("sub_df und ratios müssen die gleiche Länge haben")
    
    total_C = 0.0
    total_H = 0.0
    
    for formula, weight in zip(sub_df['formula'], ratios):
        c_atoms, h_atoms = parse_formula(formula)
        total_C += weight * c_atoms
        total_H += weight * h_atoms
    
    return total_C / total_H if total_C > 0 else np.nan

def calculate_nonlinear_viscosity(sub_df, ratios):
    total_viscosity = 0
    for density, viscosity,weight in zip(sub_df['density'], sub_df['viscosity'],ratios):
        total_viscosity += weight* viscosity/density*1e3
    return total_viscosity

def calculate_mixture_properties(sub_df, ratios, linear_props):
    """
    Berechnet die Eigenschaften einer Mischung.

    Parameters:
    - sub_df: DataFrame mit den ausgewählten Components
    - ratios: numpy array der Gewichte/Massenanteile (normalisiert)
    - linear_props: Liste der Eigenschaften, die linear gemischt werden (inkl. 'CN' optional)

    Returns:
    - predicted: numpy array der Mischwerte in der Reihenfolge von linear_props
    """
    ratios = np.array(ratios, dtype=float)
    s = np.sum(ratios)
    if s <= 0:
        raise ValueError("Summe der Anteile muss > 0 sein")
    x_norm = ratios / s

    # --- Linear berechnen (alles außer CN und CHratio) ---
    has_CN = "CN" in linear_props and "CN" in sub_df.columns
    has_CH = "CHRatio" in linear_props and "formula" in sub_df.columns
    has_viscosity = "viscosity" in linear_props and "viscosity" in sub_df.columns

    # Alle linearen Eigenschaften außer CN und CHratio
    drop_cols = []
    if has_CN:
        drop_cols.append("CN")
    if has_CH:
        drop_cols.append("CHRatio")
    if has_viscosity:
        drop_cols.append("viscosity")

    if drop_cols:
        A_linear = sub_df[linear_props].drop(columns=drop_cols).to_numpy().T
    else:
        A_linear = sub_df[linear_props].to_numpy().T

    # Lineare Mischung berechnen
    print("SubDatabank: ", sub_df, "\n normalizedRatios: ", x_norm)
    predicted_linear = A_linear @ x_norm

    predicted = np.zeros(len(linear_props))

    # Linear properties ohne CN/CHRatio
    linear_indices = [linear_props.index(p) for p in linear_props if p not in ["CN", "CHRatio", "viscosity"]]
    predicted[linear_indices] = predicted_linear
    if has_CN:
        predicted[linear_props.index("CN")] = calculate_nonlinear_cetane_number(sub_df, x_norm)


    # --- CHratio separat berechnen ---
    if has_CH:
        predicted[linear_props.index("CHRatio")] = calculate_nonlinear_ch_ratio(sub_df, x_norm)

    if has_viscosity:
        predicted[linear_props.index("viscosity")] = calculate_nonlinear_viscosity(sub_df, x_norm)
    return predicted