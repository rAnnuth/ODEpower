#%%
import itertools, unittest
from tests.base import MatlabModelTestCase
from ODEpower.ODEpower import ODEpower
from ODEpower.config import settings # Change this

from ODEpower.components_electric import *
from ODEpower.components_control import *

# exhaustive parameter grid for this model ------------------------
PARAM_SPACE = {
    "R"   : [1e-3,100],
    "L"   : [1e-5,1],
    "v_in" : [1,2e3],
    "Tsim" : [1e-5],
    "dtMax": [1e-10],
    "tolWindow": [1e-7],
}

class Test_loadRL(MatlabModelTestCase):

    @classmethod
    def setUpClass(cls):
        # expensive one-off initialisation goes here
        cls.model = "loadRL"

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
                grid.add_node(loadRL(1,{
                    "L": p['L'],
                    "R": p['R']
                }))
                grid.set_input(['v_in_1'])

                grid.set_input_values(
                    np.array([p['v_in']]),
                    np.array([2*p['v_in']]),
                    {'Tsim':p['Tsim'],'Tstep':p['Tsim']/2},
                    show = False
                )
                grid.odes_lamdify()
                grid.sim_ode(max_step=p['dtMax'],rtol=1e-8)
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
                                                tolWindow=True
                                                )


if __name__ == '__main__':
    unittest.main()
# %%
