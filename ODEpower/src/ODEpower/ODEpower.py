#%%
import os 
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

from ODEpower.ODEgraph import *
from ODEpower.ODEtool import *
from ODEpower.ODEsimulation import *
from ODEpower.ODEplot import *
from ODEpower.ODEmatlab import *


class ODEpower(ODEgraph, ODEtool, ODEsimulation, ODEplot):
    """
    Main class for the ODEpower package, providing a unified interface for modeling, simulating, and analyzing ODE-based power system networks.
    Inherits from ODEgraph, ODEtool, ODEsimulation, and ODEplot to combine graph construction, mathematical tools, simulation, and plotting.
    Optionally attaches MATLAB backend functionality if enabled in the configuration.
    """
    def __init__(self, config=None, *args, **kwargs):
        """
        Initialize the ODEpower system.

        Args:
            config: Configuration object containing simulation and backend options.
            *args, **kwargs: Additional arguments passed to parent classes.
        """
        if config == None:
            import ODEpower.config
            config = ODEpower.config.settings()
        # Step 1: Store debug flag and configuration for later use
        self.DEBUG = config.DEBUG
        self.LOG = config.LOG
        self.config = config

        # Step 2: Initialize all parent classes (ODEgraph, ODEtool, ODEsimulation, ODEplot)
        super().__init__(*args, **kwargs)

        # Step 3: Attach MATLAB backend if enabled in config
        if config.matlab_enable:
            # Create an ODEmatlab instance and attach to self.mat
            self.mat = ODEmatlab(self)