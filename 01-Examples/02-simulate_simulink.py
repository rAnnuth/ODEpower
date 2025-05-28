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

#%%
forceSym= 0

# Initial Reset
grid.graph_reset()

# Add component nodes
grid.add_node(VsourceR(1,{
    "R": 1,
},forceSymbol=forceSym))

grid.add_node(dabGAM(2,{
    "fs": 5e3,
    "N": 1,
    "Lt": 3.0420e-5,
    "Rt": 1e-3,
    "Cin": 1e-4,
    "Cout": 1e-4,
    "Goff": 1e-6, # Simulink
    "Rds": 1e-6,  # Simulink
    "Vf": 1e-3,   # Simulink
},forceSymbol=forceSym))

grid.add_node(piLine(3,{
    "R": 1,
    "L": 277e-6,
    "C": 100e-9,
    "R_c": 1e-6,
    "Len": 1
},forceSymbol=forceSym))

grid.add_node(loadVarRL(4,{
    "L": 1e-3 ,
},forceSymbol=forceSym))

# Add edges (connections) between nodes
grid.add_edge(1, 2) 
grid.add_edge(2, 3)
grid.add_edge(3, 4)

# Plot the electrical grid
grid.plot()

# Set input
grid.set_input(['v_in_1','R_4','d_2'])
#%%
# Print equations
grid.print_pretty(key='u')

#%%
grid.set_input_values(
    np.array([3900, 90, 60]),
    np.array([3900, 90 * .95, 60]),
    {'Tsim':1e-2,'Tstep':1e-5}
)


#%%
# We can see the calculated OP is not accurate and the ODE system deviates before the load step
grid.odes_lamdify()
grid.get_op()
grid.sim_ode(setOp=False)
#grid.sim_ss(Tsim=1e-4)


#%%
grid.mat.set_input()
#%%
grid.mat.set_model('m02_simulate')

#%%
grid.mat.sim_simulink()

#%%
grid.plot_states(states=['v_Cin_2'])
#%%
grid.plot_eig()
grid.plot_pf()