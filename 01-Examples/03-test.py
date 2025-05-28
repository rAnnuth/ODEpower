#%%
%reload_ext autoreload
%autoreload 2
from ODEpower.ODEpower import ODEpower
from ODEpower.config import settings
import matplotlib.pyplot as plt

from components.components_electric import *
from components.components_control import *
DEBUG = True

def _compare_with_reference( t_ode, r_ode, t_simulink, r_simulink, key, params):
    """
    Compare two timeseries (result, golden) with possibly different sampling times (t_result, t_golden).
    Resamples both to a common time base (union of all time points, sorted) using linear interpolation.
    Assumes result and golden are arrays of shape (N, ...) and t_result, t_golden are 1D arrays.
    """
    atol = 1e-2
    rtol = 1e-10
    from scipy.interpolate import interp1d

    # Create a common time base (union of all time points)
    t_common = np.union1d(t_ode, t_simulink)

    # Interpolate both series to the common time base
    interp_ode = interp1d(t_ode.flatten(), r_ode.flatten(), axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_simulink = interp1d(t_simulink.flatten(), r_simulink.flatten(), axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')

    ode_resampled = interp_ode(t_common)
    simulink_resampled = interp_simulink(t_common)

    stamp = "_".join(f"{k}-{v}" for k, v in params.items())

    def plot_debug():
        plt.plot(t_common, ode_resampled, label='ODE')
        plt.plot(t_common, simulink_resampled, label='Simulink')
        plt.legend()
        plt.title(f"Comparison for {key}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()

    # Always plot if DEBUG is True
    if DEBUG:
        plot_debug()

    np.testing.assert_allclose(
        ode_resampled, simulink_resampled,
        rtol=rtol, atol=atol,
        err_msg=f"({key}) diverged for \n{stamp}\n"
    )



#%%
model = "PI"
grid = ODEpower(settings)
#%%
grid.graph_reset()

grid.add_node(PI(1,sp.Symbol('meas_1'),sp.Symbol('out_1'),{
    "Kp": 1,
    "Ki": 1e3,
    "Td": .1,
}))


from collections import namedtuple
u = grid.graph.nodes(data=True)[1]['u']
inputs = namedtuple('inputs', u._fields + ('meas',))
grid.graph.nodes(data=True)[1]['u'] =  inputs(*u, sp.Symbol('meas_1'))


#%%
grid.set_input(['ref_1','meas_1'])
grid.set_output(grid.graph.nodes(data=True)[1]['law'])

grid.set_input_values(
    np.array([1,2]),
    np.array([2*1,2]),
    {'Tsim':1e-3,'Tstep':1e-3/2},
    show = False
)
grid.odes_lamdify()
#%%
grid.sim_ode(max_step=1e-8)
#%%
grid.mat.eval(f'addpath("/home/cao2851/git/ODEsim/_ODEpower/tests/model");')

grid.mat.set_input({"dtMax":1e-8})
grid.mat.set_model(model)

grid.mat.sim_simulink()

p = {1:'1',2:'2'}

#%%
for key in ['out_1']:
    _compare_with_reference(grid.sol_ode.t,
                                grid.sol_ode.y[key],
                                grid.sol_simulink.t[key],
                                grid.sol_simulink.x[key],
                                key,
                                p,
                                )
# %%
