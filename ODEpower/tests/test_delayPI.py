#%%
import itertools, unittest
from tests.base import MatlabModelTestCase
from ODEpower.ODEpower import ODEpower
from ODEpower.config import settings # Change this

from ODEpower.components_electric import *
from ODEpower.components_control import *

# exhaustive parameter grid for this model ------------------------
PARAM_SPACE = {
    "Kp" : [0,10],
    "Ki"   : [1e-3, 1, 1e3],
    "Td"   : [1e-6, 1e3],
    "meas" : [1,2],
    "ref" : [-.9,1],
    "Tsim" : [1e-3],
    "dtMax": [1e-7],
    "tolWindow": [1e-6],
}

class Test_delayPI(MatlabModelTestCase):

    @classmethod
    def setUpClass(cls):
        # expensive one-off initialisation goes here
        cls.model = "delayPI"

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
                grid.add_node(delayPI(1,sp.Symbol('meas_1'),sp.Symbol('out_1'),{
                    "Kp": p['Kp'],
                    "Ki": p['Ki'],
                    "Td": p['Td'],
                }))

                from collections import namedtuple
                u = grid.graph.nodes(data=True)[1]['u']
                inputs = namedtuple('inputs', u._fields + ('meas',))
                grid.graph.nodes(data=True)[1]['u'] =  inputs(*u, sp.Symbol('meas_1'))

                grid.set_input(['ref_1','meas_1'])
                grid.set_output(grid.graph.nodes(data=True)[1]['law'])

                grid.set_input_values(
                    np.array([p['ref'],p['meas']]),
                    np.array([2*p['ref'],p['meas']]),
                    {'Tsim':p['Tsim'],'Tstep':p['Tsim']/2},
                    show = False
                )
                grid.odes_lamdify()
                grid.sim_ode(max_step=p['dtMax'])
                grid.mat.eval(f'addpath("{self.model_path}");')

                grid.mat.set_input({"dtMax":p['dtMax']})
                grid.mat.set_model(self.model)

                grid.mat.sim_simulink()

                for key in ['out_1']:
                    self._compare_with_reference(grid.sol_ode.t,
                                                grid.sol_ode.y[key],
                                                grid.sol_simulink.t[key],
                                                grid.sol_simulink.x[key],
                                                key,
                                                p,
                                                tolWindow=True
                                                )


if __name__ == '__main__':
    unittest.main()
# %%
