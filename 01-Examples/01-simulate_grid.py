"""
Example 02: Simulating a Grid with Numerical Parameters
======================================================

This example demonstrates how to:
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
from ODEpower.ODEpower import ODEpower
from ODEpower.config import settings
from ODEpower.components_electric import *
from ODEpower.components_control import *
import numpy as np

# Step 1: Create the ODEpower grid object
# Initialize the grid with default settings
grid = ODEpower(settings)

# Step 2: Set forceSym=0 to use numerical (not symbolic) parameters
forceSym = 0

# Reset the graph to ensure a clean state
grid.graph_reset()

# Step 3: Add component nodes with explicit parameters
# Add a voltage source with resistance
grid.add_node(VsourceR(1, {
    "R": 1e-3,         # Series resistance [Ohm]
}, forceSymbol=forceSym))

# Add a Dual Active Bridge (DAB) converter
grid.add_node(dabGAM(2, {
    "fs": 5e3,         # Switching frequency [Hz]
    "N": 3900/30000,   # Transformer turns ratio
    "Lt": 3.0420e-5,   # Leakage inductance [H]
    "Rt": 1e-3,        # Transformer resistance [Ohm]    
    "Cin": 2.8444e-5,  # Input capacitance [F]
    "Cout": 2.8444e-4, # Output capacitance [F]
}, forceSymbol=forceSym))

# Add a pi-line model
grid.add_node(piLine(3, {
    "R": 1,        # Line resistance [Ohm]
    "L": 277e-6,   # Line inductance [H]
    "C": 100e-9,   # Line capacitance [F]
    "R_c": 1e-6,   # Capacitance loss [Ohm]
    "Len": 1       # Line length [unitless]
}, forceSymbol=forceSym))

# Add a variable RL load
grid.add_node(loadVarRL(4, {
    "L": 1e-3,     # Load inductance [H]
}, forceSymbol=forceSym))

# Step 4: Connect the components
# Define connections between the nodes
grid.add_edge(1, 2)
grid.add_edge(2, 3)
grid.add_edge(3, 4)

# Step 5: Visualize the electrical grid structure
# Plot the grid to verify the structure
grid.plot()

# Step 6: Set input variable names (must match expected component inputs)
grid.set_input(['v_in_1', 'R_4', 'd_2'])

# Step 7: Print the mapping of input variables (for reference)
grid.print_pretty(key='u')

# Step 8: Set input values for simulation
# Define initial and step values for the inputs
grid.set_input_values(
    np.array([3900, 90, 30e3]),         # Initial values
    np.array([3900, 90 * 0.95, 30e3]), # Step values (e.g., 5% load drop)
    {'Tsim': 1e-4, 'Tstep': 1e-5}      # Simulation time and step
)

# Step 9: Run ODE and steady-state simulations
# Generate ODE functions and simulate the system
grid.odes_lamdify()
grid.get_op()
grid.sim_ode(setOp=True, Tsim=1e-4)
grid.sim_ss(Tsim=1e-4)

# Step 10: Plot the state variable(s) of interest
# Example: Plot inductor current at node 3
grid.plot_states(states=['i_L_3'])

# Plot eigenvalues and participation factors for system analysis
grid.plot_eig()
grid.plot_pf()
