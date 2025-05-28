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
model = "dabGAM"
grid = ODEpower(settings)
#%%
_d = np.array([])
_simulink = np.array([])
_ode = np.array([])
for i in np.linspace(3,90,3):
    print(i)
    p = {
        "fs"   : 50000,
        "Cin"   : 1e-5,
        "Cout"   : 1e-3,
        "Rt"   : 1e-3,
        "Lt"   : 1e-5,
        "N"   : 11,
        "R"   : 1e-3,
        "RL"   : 10,
        "LL"   : 1e-3,
        "v_in" : 50,
        "d" : i,
        #"Tsim" : 7e-2,
        "Tsim" : 4e-2,
        "dtMax": 1e-7,
        #Simulink
        "Rds"   : 1e-6,
        "Goff"   : 1e-8,
        "Vf"   : 1e-5,
        "tolWindow": 10e-7,
    }

    grid.graph_reset()
    grid.add_node(VsourceR(1,{
        "R": p['R'],
    }))
    grid.add_node(dabGAM(2,{
        "fs": p['fs'],
        "Cin": p['Cin'],
        "Cout": p['Cout'],
        "Lt": p['Lt'],
        "Rt": p['Rt'],
        "N": p['N'],
        "Rds"  : p['Rds'],
        "Goff" : p['Goff'],
        "Vf"   : p['Vf'],
        "order": 5,
        "RL"  : p['RL'],
    }))
    grid.add_node(loadRL(3,{
        "R"  : p['RL'],
        "L" : p['LL'],
    }))
    grid.add_edge(1,2)
    grid.add_edge(2,3)

    grid.set_input(['v_in_1','d_2'])

    grid.set_input_values(
        np.array([p['v_in'],p['d']]),
        #np.array([2*p['v_in'],p['d']]),
        np.array([p['v_in'],p['d']]),
        {'Tsim':p['Tsim'],'Tstep':p['Tsim']/2},
        show = False
    )

    grid.odes_lamdify()
    grid.sim_ode(max_step=p['dtMax'])
    grid.mat.eval(f'addpath("/home/cao2851/git/ODEsim/_ODEpower/tests/model");')

    grid.mat.set_input({"dtMax":p['dtMax']})
    grid.mat.set_model(model)

    grid.mat.sim_simulink()

    _ode = np.append(_ode, grid.sol_ode.x['i_L_3'][-500:].mean())
    _simulink = np.append(_simulink, grid.sol_simulink.x['i_L_3'][-500:].mean())
    _d = np.append(_d,i)

#%%
# N = .1
atol = 5e-2
rtol = 5e-2

atol = 5e-1
rtol = 5e-1
key = ''
np.testing.assert_allclose(
    _ode, _simulink,
    rtol=rtol, atol=atol,
    err_msg=f"({key}) diverged for \n"
)


#%%
plt.plot(_d, _ode, label='ODE')
plt.plot(_d, _simulink, label='Simulink')
plt.legend()


#%%

grid.plot_states(states=['v_Cin_2'])
#%%
grid.plot_states(states=['v_Cout_2'])
#%%
grid.plot_states(states=['i_L_3'])
#%%
plt.plot(grid.sol_simulink.x['i_L_3'][-500:])
#%%
grid.sol_simulink.x['i_L_3'][-500:].mean()
#%%
grid.sol_ode.x['i_L_3'][-500:].mean()
#%%
#%
x = np.array([])
#%%
x


#%%
np.linspace(10,90,10)

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
#_ode[-1] = 1.325
atol = 5e-2
rtol = 5e-2
key = ''
np.testing.assert_allclose(
    _ode, _simulink,
    rtol=rtol, atol=atol,
    err_msg=f"({key}) diverged for \n"
)