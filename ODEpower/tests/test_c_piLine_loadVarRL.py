#%%
import itertools, unittest
from tests.base import MatlabModelTestCase
from ODEpower.ODEpower import ODEpower
from ODEpower.config import settings # Change this

from ODEpower.components_electric import *
from ODEpower.components_control import *

# exhaustive parameter grid for this model ------------------------
PARAM_SPACE = {
    "R"   : [100],
    #"R"   : [1e-9,100],
    "L"   : [1],
    #"L"   : [1e-9,1],
    "Lload"   : [1e-9,1],
    #"C"   : [1e-9,1],
    "C"   : [1],
    #"R_c"   : [1e-6,50],
    "R_c"   : [50],
    #"Len" : [1e-3,1e5],
    "Len" : [1e5],
    "i_in" : [1,2],
    #"i_out" : [.2,.9],
    "R_2" : [.9,1],
    "Tsim" : [1e-3],
    "dtMax": [1e-6]
}

class Test_piLine_loadVarR(MatlabModelTestCase):

    @classmethod
    def setUpClass(cls):
        # expensive one-off initialisation goes here
        cls.model = "c_piLine_loadVarRL"

    def test_parameter_grid(self):
        keys, values = zip(*PARAM_SPACE.items())
        grid = ODEpower(settings)
        grid.DEBUG = False 
        self.grid = grid
        for combo in itertools.product(*values):
            p = dict(zip(keys, combo))

            # subTest => each combo is reported separately
            with self.subTest(**p):
                grid.graph_reset()
                grid.add_node(piLine(1,{
                    "R": p['R'],
                    "L": p['L'],
                    "C": p['C'],
                    "R_c": p['R_c'],
                    "Len": p['Len']
                }))
                grid.add_node(loadVarRL(2,{
                   "L": p['Lload'] 
                }))
                grid.add_edge(1,2)

                grid.set_input(['i_in_1','R_2'])

                grid.set_input_values(
                    np.array([p['i_in'],p['R_2']]),
                    np.array([2*p['i_in'],p['R_2']]),
                    {'Tsim':p['Tsim'],'Tstep':p['Tsim']/2},
                    show = False
                )
                grid.odes_lamdify()
                grid.sim_ode(max_step=p['dtMax'])
                grid.mat.eval(f'addpath("{self.model_path}");')

                grid.mat.set_input({"dtMax":p['dtMax']})
                grid.mat.set_model(self.model)

                grid.mat.sim_simulink()

                for key in grid.sol_ode.x.keys():
                    self._compare_with_reference(grid.sol_ode.t,
                                                grid.sol_ode.x[key],
                                                grid.sol_simulink.t[key],
                                                grid.sol_simulink.x[key],
                                                key,
                                                p,
                                                )


if __name__ == '__main__':
    unittest.main()
# %%
