#%%
# Import necessary modules
from ODEpower.ODEpower import ODEpower
from ODEpower.config import settings
from ODEpower.components_electric import *
from ODEpower.components_control import *
import numpy as np

#%%
#%%
# 1. Create the ODEpower grid object
# Initialize the grid object with the configuration settings
grid = ODEpower(settings)

# 2. Set forceSym=0 to use numerical (not symbolic) parameters
# We could skip this and per default the argument is 0
forceSym = 0

#%%
# Reset the grid graph to start fresh
grid.graph_reset()

grid.add_node(VsourceR(1, {
    "R": 1e-3,         # Series resistance [Ohm]
}, forceSymbol=forceSym))

# Add a Dual Active Bridge (DAB) component with specific parameters
grid.add_node(dabGAM(2, {
    "fs": 5e3,         # Switching frequency [Hz]
    "N": 3900/30000,   # Transformer turns ratio
    "Lt": 3.0420e-5,   # Leakage inductance [H]
    "Rt": 1e-3,        # Transformer resistance [Ohm]
    "Cin": 2.8444e-5,  # Input capacitance [F]
    "Cout": 2.8444e-4, # Output capacitance [F]
}, forceSymbol=forceSym))

grid.add_edge(1,2)


#%%
# We can access the graph and check nodes and edges
grid.graph.nodes(data=True)
grid.graph.edges(data=True)

#%%
# We can access the graph and check nodes and edges
grid.graph.nodes(data=True)
grid.graph.edges(data=True)

#%%
# We could replace the None value of this to pass a valid matlab engine object
print(settings.matlab_engine)

#%%
# We could replace the None value of this to pass a valid matlab engine object
print(settings.matlab_engine)

#%%
# Printing is also supported / key: One of 'u', 'x', 'odes', 'algebraic', 'eig', 'op'.
grid.aggregate()
grid.print_pretty('x')

#%%
# The package also supports parametric investigations, just define a sympy symbol as parameter and use the function 
# TODO add example

grid.parametric_ode