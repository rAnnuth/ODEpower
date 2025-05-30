# Getting Started

This guide will help you get started with the ODEpower package.

## Installation

```bash
pip install odepower
```

## Usage Example

```python
from ODEpower.ODEpower import ODEpower
from ODEpower.config import Config

config = Config()
model = ODEpower(config)
# Add nodes, edges, run simulations, etc.
```

A local custom configuration file should be used. The default content is:

```python
from dataclasses import dataclass

@dataclass
class settings:
    """ODEpower Settings"""
    DEBUG: bool = False
    LOG: bool = False 
    pySolver: str = 'LSODA' 
    casadiEnable: bool = True
    casadiPath: str = ''
    matlab_engine = None
    matlab_enable: bool = True 
    matlab_model_path: str = ''
```

The settings.matlab_engine can be replaced by a existing engine. Otherwise ODEpower connects to an existing session or starts Matlab.

See the API Reference for details on each class and method.
