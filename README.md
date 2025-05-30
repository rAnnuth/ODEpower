
# ODEsim

ODEpower is a Python package for modeling, simulating, and analyzing ODE-based power system grid. It provides a unified interface for graph construction, mathematical tools, simulation, plotting, and MATLAB backend integration.

## Features

- Integrated DC ODE models
- Integrated AC ODE models
- Modular component-based modeling for power electronics and control
- Stability, interaction and modal analysis
- Time-domain simulation of systems
- Tools for reading and processing MATLAB Simulink Simulation
- Extensive test suite for model validation

## Installation

```bash
pip install ODEpower
```
or, for development:
```bash
git clone https://github.com/yourusername/ODEsim.git
cd ODEpower
pip install -e .
```

## Usage

```python
from ODEpower.ODEpower.tool import read_mat_script, map_nested_dicts

# Read variables from a MATLAB script
variables = read_mat_script('params.m')

# Apply a function to all values in a nested dict
result = map_nested_dicts(variables, lambda x: x)
```

## Documentation

Full documentation is available at [https://odepower.readthedocs.io](https://odepower.readthedocs.io)

To build the docs locally (with the requirements of ODEpower/requirements.txt):
```bash
cd ODEpower
mkdocs serve
```

## Contributing

Contributions are welcome! Please open issues or pull requests on GitHub.

## License

This project is licensed under the GNU v3 License. See `LICENSE.txt` for details.
