
# ODEsim

ODEsim is a Python package for simulating and analyzing ordinary differential equation (ODE) models, with a focus on power systems and AC ODE models.

## Features

- Integrated AC ODE models
- Tools for reading and processing MATLAB scripts
- Utilities for working with nested dictionaries
- Modular component-based modeling for power electronics and control
- Extensive test suite for model validation

## Installation

```bash
pip install .
```
or, for development:
```bash
git clone https://github.com/yourusername/ODEsim.git
cd ODEsim
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

Full documentation is available at [https://odesim.readthedocs.io](https://odesim.readthedocs.io) (after setup).

To build the docs locally:
```bash
cd ODEpower
mkdocs serve
```

## Contributing

Contributions are welcome! Please open issues or pull requests on GitHub.

## License

This project is licensed under the GNU v3 License. See `LICENSE.txt` for details.
