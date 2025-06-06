#%%
import itertools, unittest
from tests.base import MatlabModelTestCase
from ODEpower.ODEpower import ODEpower
from ODEpower.config import settings # Change this

from ODEpower.components_electric import *
from ODEpower.components_control import *

# exhaustive parameter grid for this model ------------------------
PARAM_SPACE = {
    "Kp" : [1,10],
    "Ki"   : [1e-3,1e5],
    "Td"   : [1e-6, 1e3],
    "Fb"   : [1e-3,1e3],
    "Fb_d" : [2e-3,2e3],
    "K_d"  : [.4,10],
    "P_d"  : [.1,100],
    "meas" : [1,2],
    "i_meas": [3,-2.4],
    "ref"  : [-.9,1],
    "Tsim" : [1e-6],
    "dtMax": [1e-10],
    "tolWindow": [1e-8],
}

class Test_delayLpPIDroop(MatlabModelTestCase):

    @classmethod
    def setUpClass(cls):
        # expensive one-off initialisation goes here
        cls.model = "delayLpPIDroop"

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
                grid.add_node(delayLpPIDroop(1,[sp.Symbol('meas_1'),sp.Symbol('i_meas_1')],sp.Symbol('out_1'),{
                    "Kp": p['Kp'],
                    "Ki": p['Ki'],
                    "Fb": p['Fb'],
                    "Fb_d": p['Fb_d'],
                    "K_d": p['K_d'],
                    "P_d": p['P_d'],
                    "Td": p['Td'],
                }))

                from collections import namedtuple
                u = grid.graph.nodes(data=True)[1]['u']
                inputs = namedtuple('inputs', u._fields + ('meas',) + ('i_meas',))
                grid.graph.nodes(data=True)[1]['u'] =  inputs(*u, sp.Symbol('meas_1'), sp.Symbol('i_meas_1'))

                grid.set_input(['ref_1','meas_1','i_meas_1'])
                grid.set_output(grid.graph.nodes(data=True)[1]['law'])

                grid.set_input_values(
                    np.array([p['ref'],p['meas'],p['i_meas']]),
                    np.array([2*p['ref'],p['meas'],p['i_meas']]),
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
