#%%
import itertools, unittest
from tests.base import MatlabModelTestCase
from ODEpower.ODEpower import ODEpower
from ODEpower.config import settings # Change this

from ODEpower.components_electric import *
from ODEpower.components_control import *

# exhaustive parameter grid for this model ------------------------
PARAM_SPACE1 = {
        "fs"   : [50000],
        "Cin"   : [1e-5],
        "Cout"   : [5e-3],
        "Rt"   : [1e-3,10],
        "Lt"   : [1e-5],
        "N"   : [11],
        "R"   : [1e-3],
        "RL"   : [10],
        "LL"   : [1e-3],
        "v_in" : [50],
        #"Tsim" : 7e-2,
        "Tsim" : [7e-2],
        "dtMax": [1e-7],
        #Simulink
        "Rds"   : [1e-6],
        "Goff"   : [1e-8],
        "Vf"   : [1e-5],
}

PARAM_SPACE2 = {
        "fs"   : [50000],
        "Cin"   : [1e-5],
        "Cout"   : [5e-4],
        "Rt"   : [1e-3,10],
        "Lt"   : [1e-5],
        "N"   : [.11],
        "R"   : [1e-3],
        "RL"   : [10],
        "LL"   : [1e-3],
        "v_in" : [50],
        #"Tsim" : 7e-2,
        "Tsim" : [7e-2],
        "dtMax": [1e-7],
        #Simulink
        "Rds"   : [1e-6],
        "Goff"   : [1e-8],
        "Vf"   : [1e-5],
}

class Test_dabGAM(MatlabModelTestCase):
    #TODO Maybe check more parameter? 

    @classmethod
    def setUpClass(cls):
        # expensive one-off initialisation goes here
        cls.model = "dabGAM"

    def test_high_N(self):
        keys, values = zip(*PARAM_SPACE1.items())
        grid = ODEpower(settings)
        grid.DEBUG = False 
        self.grid = grid
        for combo in itertools.product(*values):
            _simulink = np.array([])
            _ode = np.array([])
            for i in np.linspace(3,90,4):
                p = dict(zip(keys, combo))

                # subTest => each combo is reported separately
                with self.subTest(**p):
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
                    }))
                    grid.add_node(loadRL(3,{
                        "R"  : p['RL'],
                        "L" : p['LL'],
                    }))
                    grid.add_edge(1,2)
                    grid.add_edge(2,3)

                    grid.set_input(['v_in_1','d_2'])

                    grid.set_input_values(
                        np.array([p['v_in'],i]),
                        np.array([p['v_in'],i]),
                        {'Tsim':p['Tsim'],'Tstep':p['Tsim']/2},
                        show = False
                    )
                    grid.odes_lamdify()
                    grid.sim_ode(max_step=p['dtMax'])
                    grid.mat.eval(f'addpath("{self.model_path}");')

                    grid.mat.set_input({"dtMax":p['dtMax']})
                    grid.mat.set_model(self.model)

                    grid.mat.sim_simulink()

                    _ode = np.append(_ode, grid.sol_ode.x['i_L_3'][-500:].mean())
                    _simulink = np.append(_simulink, grid.sol_simulink.x['i_L_3'][-500:].mean())

                    atol = 5e-1
                    rtol = 5e-1
                    key = ''
                    np.testing.assert_allclose(
                        _ode, _simulink,
                        rtol=rtol, atol=atol,
                        err_msg=f"({key}) diverged for \n"
                    )

    def test_low_N(self):
        keys, values = zip(*PARAM_SPACE2.items())
        grid = ODEpower(settings)
        grid.DEBUG = False 
        self.grid = grid
        for combo in itertools.product(*values):
            _simulink = np.array([])
            _ode = np.array([])
            for i in np.linspace(3,90,4):
                p = dict(zip(keys, combo))

                # subTest => each combo is reported separately
                with self.subTest(**p):
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
                    }))
                    grid.add_node(loadRL(3,{
                        "R"  : p['RL'],
                        "L" : p['LL'],
                    }))
                    grid.add_edge(1,2)
                    grid.add_edge(2,3)

                    grid.set_input(['v_in_1','d_2'])

                    grid.set_input_values(
                        np.array([p['v_in'],i]),
                        np.array([p['v_in'],i]),
                        {'Tsim':p['Tsim'],'Tstep':p['Tsim']/2},
                        show = False
                    )
                    grid.odes_lamdify()
                    grid.sim_ode(max_step=p['dtMax'])
                    grid.mat.eval(f'addpath("{self.model_path}");')

                    grid.mat.set_input({"dtMax":p['dtMax']})
                    grid.mat.set_model(self.model)

                    grid.mat.sim_simulink()

                    _ode = np.append(_ode, grid.sol_ode.x['i_L_3'][-500:].mean())
                    _simulink = np.append(_simulink, grid.sol_simulink.x['i_L_3'][-500:].mean())

                    atol = 5e-1
                    rtol = 5e-1
                    key = ''
                    np.testing.assert_allclose(
                        _ode, _simulink,
                        rtol=rtol, atol=atol,
                        err_msg=f"({key}) diverged for \n"
                    )


if __name__ == '__main__':
    unittest.main()
# %%
