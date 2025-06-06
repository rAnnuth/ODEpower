#%%
import itertools, unittest
from tests.base import MatlabModelTestCase
from ODEpower.ODEpower import ODEpower
from ODEpower.config import settings # Change this

from ODEpower.components_electric import *
from ODEpower.components_control import *

# exhaustive parameter grid for this model ------------------------
PARAM_SPACE = {
    "R"   : [30,100],
    "R1"   : [20,300],
    "R2"   : [10,500],
    #"R"   : [1e-9,100],
    "L"   : [1,.1],
    #"L"   : [1e-9,1],
    #"C"   : [1e-9,1],
    "C"   : [1],
    "R_c"   : [1e-5,500],
    "Len" : [1e-3,1e5],
    "i_in" : [1,2],
    "i_in1" : [1.5,2.5],
    "i_out" : [.9,1],
    "Tsim" : [1e-4],
    "dtMax": [1e-6]
}

class Test_KCL(MatlabModelTestCase):

    @classmethod
    def setUpClass(cls):
        # expensive one-off initialisation goes here
        cls.model = "c_KCL"

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

                grid.add_node(piLine(2,{
                    "R": p['R1'],
                    "L": p['L'],
                    "C": p['C'],
                    "R_c": p['R_c'],
                    "Len": p['Len']
                }))

                grid.add_node(piLine(3,{
                    "R": p['R2'],
                    "L": p['L'],
                    "C": p['C'],
                    "R_c": p['R_c'],
                    "Len": p['Len']
                }))

                grid.add_edge(1,3)
                grid.add_edge(2,3)
                grid.add_kcl(3)

                grid.set_input(['i_in_1','i_in_2','i_out_3'])

                grid.set_input_values(
                    np.array([p['i_in'],p['i_in1'],p['i_out']]),
                    np.array([2*p['i_in'],3*p['i_in1'],p['i_out']]),
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
