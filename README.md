# Many-Objective Optimization Tool for Input Formulation for Fuel Surrogates

This is a Python program that summarizes the modules of numerical calculations fuel behaviour (Phase Diagram and Distillation Curve) as well as the additive properties for fuel emulation into one optimization algorithm. The design of this algorithm makes it possible to ad more objectives, e.g. physical properties (flame speed, etc.)

## Overview
This project is structured in 3 parts. 
| Module | Description |
|--------|-------------|
| **Optimization loop** | Here the logistics and the logic of the optimization are handled |
| **Additive Characteristics** | Functions that calculate the fuel properties are called, located in subfunctions of the optimization loop |
| **Equilibrium Calculations** | Two libraries that will have to be installed for first execution of the code |

The module **Additive Characteristics** was developed in Python by Amelie Dimke as part of her M.Sc. Aerospace Engineering studies. The **Equilibrium Calculations** were developed in Fortran by Dr. Anton Zizin as part of his PhD at the German Aerospace Center and translated by Alvaro Piccini has part of his B.Sc. Aerospace Engineering studies. The optimization loop was constructed as a Master thesis by Sascha Märtin as part of his studies of Aerospace Engineering. 

## Structure
```
masterthesis/
├── Data
│   ├── 2PD                 # Data for the equilibrium calculations
│   ├── blend               # Data for calculating the additive properties
│   └── physDatabank.csv    # Database of molecules
├── phase                   # Equilibrium solvers 
│   ├── DistillationCurve   
│   └── tumkin-phase
├── utils
├── algorithms.py
└── main.py                 # Main python file to start simulation

```
## Requirements
- python >3.7
- numpy
- deap
- sklearn
- two_phase_lib
- distillation_lib

For installation of the "*_lib"- modules navigate to "./phase/DistillationCurve/distillation" or "./phase/tumkin-phase/two-phase" and in the commandline execute:
```bash
    pip install -e .
```

## Usage
- Navigate to location of "main.py". 
- Open command line
- "python main.py" and follow instructions




## Citation

If you use this library in academic work, please cite:

- **Dr. Anton Zizin** — Original Fortran implementation (DLR, 2012)
- **Álvaro Piccini** — Python translation (TUM, 2025)
- **Amelie Dimke** — Additive Properties (TUM,2025)
- **Sascha Märtin** — Optimization Loop (TUM, 2026)
- **Dr. Slavinskaya** - Supervision