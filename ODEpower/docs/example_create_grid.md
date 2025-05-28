"""
Example 02: Simulink Co-Simulation with ODEpower
================================================

This example demonstrates how to:
- Build a grid with ODEpower
- Set up for co-simulation with Simulink (via MATLAB interface)
- Run ODE simulations and Simulink co-simulation
- Visualize results

Steps covered:
1. Importing ODEpower and Components
2. Creating the ODEpower grid object
3. Adding components with numerical parameters
4. Connecting components
5. Visualizing the grid
6. Setting input variable names
7. Printing input variable mapping
8. Setting input values and simulation parameters
9. Running ODE simulation
10. Preparing and running Simulink co-simulation
11. Plotting results
"""

#%%
# (Optional) Enable autoreload for interactive development (Jupyter/IPython)
%reload_ext autoreload
%autoreload 2

from ODEpower.ODEpower import ODEpower
from ODEpower.config import settings
from components.components_electric import *
from components.components_control import *
import numpy as np

#%%
# 1. Create the ODEpower grid object
grid = ODEpower(settings)

# 2. Set forceSym=0 to use numerical parameters
forceSym = 0

grid.graph_reset()

# 3. Add component nodes with explicit parameters
grid.add_node(VsourceR(1, {
    "R": 1,
}, forceSymbol=forceSym))

grid.add_node(dabGAM(2, {
    "fs": 5e3,
    "N": 1,
    "Lt": 3.0420e-5,
    "Rt": 1e-3,
    "Cin": 1e-4,
    "Cout": 1e-4,
    "Goff": 1e-6, # Simulink-specific parameter
    "Rds": 1e-6,  # Simulink-specific parameter
    "Vf": 1e-3,   # Simulink-specific parameter
}, forceSymbol=forceSym))

grid.add_node(piLine(3, {
    "R": 1,
    "L": 277e-6,
    "C": 100e-9,
    "R_c": 1e-6,
    "Len": 1
}, forceSymbol=forceSym))

grid.add_node(loadVarRL(4, {
    "L": 1e-3,
}, forceSymbol=forceSym))

# 4. Connect the components
grid.add_edge(1, 2)
grid.add_edge(2, 3)
grid.add_edge(3, 4)

# 5. Visualize the electrical grid structure
grid.plot()

# 6. Set input variable names (must match expected component inputs)
grid.set_input(['v_in_1', 'R_4', 'd_2'])

#%%
# 7. Print the mapping of input variables (for reference)
grid.print_pretty(key='u')

#%%
# 8. Set input values for simulation
grid.set_input_values(
    np.array([3900, 90, 60]),         # Initial values
    np.array([3900, 90 * 0.95, 60]), # Step values (e.g., 5% load drop)
    {'Tsim': 1e-2, 'Tstep': 1e-5}    # Simulation time and step
)

#%%
# 9. Run ODE simulation (setOp=False: do not use calculated OP as initial state)
grid.odes_lamdify()
grid.get_op()
grid.sim_ode(setOp=False)
#grid.sim_ss(Tsim=1e-4)  # (Optional) steady-state simulation

#%%
# 10. Prepare and run Simulink co-simulation
grid.mat.set_input()
grid.mat.set_model('m02_simulate')
grid.mat.sim_simulink()

#%%
# 11. Plot state variable(s) of interest (e.g., input capacitor voltage at node 2)
grid.plot_states(states=['v_Cin_2'])

#%%
# 12. Plot eigenvalues and participation factors for system analysis
grid.plot_eig()
grid.plot_pf()
#%%
