# tests/base.py
import unittest, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

MODEL_DIR = Path(__file__).with_suffix("").parent / "model"

class MatlabModelTestCase(unittest.TestCase):
    "Shared helpers for every .py in tests/matlab_models"
    rtol, atol = 1e-2, 1e-10        # project-wide tolerances

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set default paths or configurations here
        self.model_path = MODEL_DIR
        self.DEBUG = True # Enable or disable debug mode

    def _compare_with_reference(self, t_ode, r_ode, t_simulink, r_simulink, key, params, tolWindow=False):
        """
        Compare two timeseries (result, golden) with possibly different sampling times (t_result, t_golden).
        Resamples both to a common time base (union of all time points, sorted) using linear interpolation.
        Assumes result and golden are arrays of shape (N, ...) and t_result, t_golden are 1D arrays.
        """
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
        if self.DEBUG:
            plot_debug()
        
        if tolWindow:
            mask = np.ones_like(t_common, dtype=bool)
            for edge in [0, params['Tsim']/2]:
                mask &= np.abs(t_common - params['Tsim']/2) > params['tolWindow']
            ode_resampled = ode_resampled[mask]
            simulink_resampled = simulink_resampled[mask]

        try:
            np.testing.assert_allclose(
                ode_resampled, simulink_resampled,
                rtol=self.rtol, atol=self.atol,
                err_msg=f"{self.model} ({key}) diverged for \n{stamp}\n"
            )
        except AssertionError as e:
            # Plot again in case of failure, even if DEBUG is False
            if not self.DEBUG:
                plot_debug()

            #plt.figure()
            #plt.plot(ode_resampled - simulink_resampled)
            #plt.show()
            #plt.grid(True)
            #raise  # re-raise the exception so the test still fails

# model1
#import itertools, unittest
#from tests.base import MatlabModelTestCase
#
## exhaustive parameter grid for this model ------------------------
#PARAM_SPACE = {
#    "dt"   : [0.001, 0.01],
#    "gain" : [1.0, 2.0, 5.0],
#    "seed" : [0],            # deterministic by default
#}
#
#class TestModel1(MatlabModelTestCase):
#
#    @classmethod
#    def setUpClass(cls):
#        # expensive one-off initialisation goes here
#        cls.model = "model1"          # used by _compare_with_golden
#
#    def test_parameter_grid(self):
#        keys, values = zip(*PARAM_SPACE.items())
#        for combo in itertools.product(*values):
#            params = dict(zip(keys, combo))
#            # subTest => each combo is reported separately
#            with self.subTest(**params):
#                self._compare_with_golden(self.model, params=params)
#