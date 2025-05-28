#%%
%reload_ext autoreload
%autoreload 2
from ODEpower.ODEpower import ODEpower
from ODEpower.config import settings

from components.components_electric import *
from components.components_control import *

#%%

# Create a class
grid = ODEpower(settings)
# For now we only create symbolic equations which does not require parameters
forceSym= 0

# Initial Reset
grid.graph_reset()

# Add component nodes
grid.add_node(dabFHA(1,{
    "fs": 5e3,
    "N": 3900/30000,
    "Lt": 3.0420e-5,
    "Cout": 2.8444e-4,
},forceSymbol=forceSym))

grid.add_node(piLine(2,{
    "R": 1,
    "L": 277e-6,
    "C": 100e-9,
    "R_c": 1e-6,
    "Len": 1
},forceSymbol=forceSym))
grid.add_node(loadVarRL(3,{
    "L": 1e-3 ,
},forceSymbol=forceSym))

# Add edges (connections) between nodes
grid.add_edge(1, 2) 
grid.add_edge(2, 3)

# Plot the electrical grid
grid.plot()

# Set input
grid.set_input(['v_in_1','R_3','d_1'])
#%%
# Print equations
grid.print_pretty(key='u')

#%%
grid.set_input_values(
    np.array([3900, 90, 30e3]),
    np.array([3900, 90 * .95, 30e3]),
    {'Tsim':1e-4,'Tstep':1e-5}
)


#%%
# We can see the calculated OP is not accurate and the ODE system deviates before the load step
grid.odes_lamdify()
grid.get_op()
grid.sim_ode(setOp=True,Tsim=1e-4)
grid.sim_ss(Tsim=1e-4)

grid.plot_states(states=['i_L_3'])


#%%
grid.plot_eig()
grid.plot_pf()
# %%
