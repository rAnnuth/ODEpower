"""
Configuration module for ODEpower.

This module defines the settings for the ODEpower package.

Classes:
    settings: Dataclass for storing configuration settings.
"""

from dataclasses import dataclass

@dataclass
class settings:
    """
    ODEpower Settings.

    Attributes:
        DEBUG (bool): Enable or disable debug mode.
        LOG (bool): Enable or disable logging.
        pySolver (str): Python solver to use (default: 'LSODA').
        matlab_engine: MATLAB engine instance.
        matlab_enable (bool): Enable or disable MATLAB integration.
        matlab_model_path (str): Path to the MATLAB model files.
    """
    DEBUG: bool = False
    LOG: bool = False 
    pySolver: str = 'LSODA' # ...
    matlab_engine = None
    matlab_enable: bool = True 
    matlab_model_path: str = ''