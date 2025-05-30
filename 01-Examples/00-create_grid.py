"""
Example 01: Simulating a Grid with Numerical Parameters
======================================================

This example shows how to:
- Build a grid with specific component parameters
- Set up and run time-domain simulations
- Visualize simulation results (states, eigenvalues, participation factors)

Steps covered:
1. Importing ODEpower and Components
2. Creating the ODEpower grid object
3. Adding components with numerical parameters
4. Connecting components
5. Visualizing the grid
6. Setting input variable names
7. Printing input variable mapping
8. Setting input values and simulation parameters
9. Running ODE and steady-state simulations
10. Plotting results
"""

#%%
# (Optional) Enable autoreload for interactive development (Jupyter/IPython)
%reload_ext autoreload
%autoreload 2

# Import necessary modules
from ODEpower.ODEpower import ODEpower
from ODEpower.config import settings
from components.components_electric import *
from components.components_control import *
import numpy as np

#%%
# 1. Create the ODEpower grid object
# Initialize the grid object with the configuration settings
grid = ODEpower(settings)

# 2. Set forceSym=0 to use numerical (not symbolic) parameters
forceSym = 0

# Reset the grid graph to start fresh
grid.graph_reset()

#%%
# 3. Add component nodes with explicit parameters
# Add a Dual Active Bridge (DAB) component with specific parameters
grid.add_node(dabFHA(1, {
    "fs": 5e3,         # Switching frequency [Hz]
    "N": 3900/30000,   # Transformer turns ratio
    "Lt": 3.0420e-5,   # Leakage inductance [H]
    "Cout": 2.8444e-4, # Output capacitance [F]
}, forceSymbol=forceSym))

# Add a pi-line component representing a transmission line
grid.add_node(piLine(2, {
    "R": 1,        # Line resistance [Ohm]
    "L": 277e-6,   # Line inductance [H]
    "C": 100e-9,   # Line capacitance [F]
    "R_c": 1e-6,   # Capacitance loss [Ohm]
    "Len": 1       # Line length [unitless]
}, forceSymbol=forceSym))

# Add a variable RL load component
grid.add_node(loadVarRL(3, {
    "L": 1e-3,     # Load inductance [H]
}, forceSymbol=forceSym))

#%%
# 4. Connect the components
# Define the connections between the components
grid.add_edge(1, 2)  # Connect DAB to pi-line
grid.add_edge(2, 3)  # Connect pi-line to load

#%%
# 5. Visualize the electrical grid structure
# Plot the grid structure to verify the connections
grid.plot()

#%%
# 6. Set input variable names (must match expected component inputs)
# Define the input variables for the grid
grid.set_input(['v_in_1', 'R_3', 'd_1'])

#%%
# 7. Print the mapping of input variables (for reference)
# Display the mapping of input variables to components
grid.print_pretty(key='u')

#%%
# 8. Set input values for simulation
# Define initial and step values for the input variables
# Also set simulation time and step size
grid.set_input_values(
    np.array([3900, 90, 30e3]),         # Initial values
    np.array([3900, 90 * 0.95, 30e3]), # Step values (e.g., 5% load drop)
    {'Tsim': 1e-4, 'Tstep': 1e-5}      # Simulation time and step
)

#%%
# 9. Run ODE and steady-state simulations
# Generate ODE equations and calculate operating points
grid.odes_lamdify()
grid.get_op()

# Run time-domain simulation
grid.sim_ode(setOp=True, Tsim=1e-4)

# Optionally, run steady-state simulation
grid.sim_ss(Tsim=1e-4)

#%%
# 10. Plot the state variable(s) of interest
# For example, plot the inductor current at node 3
grid.plot_states(states=['i_L_3'])

#%%
# 11. Plot eigenvalues and participation factors for system analysis
# Analyze system stability and participation factors
grid.plot_eig()
grid.plot_pf()
#%%
