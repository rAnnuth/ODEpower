# ODEsim Power System Connection Equations

This document describes the structure and usage of the `AlgebraicEquation` class, which is responsible for generating algebraic connection equations between different power system components in the ODEsim framework.

## Overview

The `AlgebraicEquation` class provides a systematic way to define the algebraic constraints that arise when connecting two components in a power system network. Each method in the class corresponds to a specific pair of component types and returns the symbolic equations (using `sympy`) that enforce the physical connection laws (e.g., current continuity, voltage matching).

The equations are written using named attributes (e.g., `x.v_Cin`, `u.i_out`) for clarity and maintainability, leveraging the namedtuple structure of component states and inputs.

## Usage

- The class is initialized with a `networkx` graph, where each node represents a component and contains its symbolic state and input variables.
- The `generate_equations` method is called with two node attribute dictionaries (each must include `name` and `id`). It dispatches to the appropriate connection method based on the component types.
- Each connection method returns a `sympy.Matrix` of algebraic equations for that connection.

## Example

```python
from components.connection import AlgebraicEquation
import networkx as nx

# Assume G is a networkx graph with nodes for each component
alg_eq = AlgebraicEquation(G)

# node1 and node2 are node attribute dicts for two connected components
eqs = alg_eq.generate_equations(node1, node2)
```

## Supported Connections

The following component pairs are supported (see the source code for the full list and details):

- piLine ↔ loadVarRL
- piLine ↔ loadVarR
- dabFHA ↔ piLine
- dabGAM ↔ piLine
- boost ↔ piLine
- buck ↔ piLine
- buck ↔ loadVarRL
- buck ↔ loadRL
- dabGAM ↔ loadVarR
- piLine ↔ dabGAM
- piLine ↔ boost
- piLine ↔ piLine
- VsourceR ↔ piLine
- VsourceR ↔ dabGAM
- VsourceR ↔ boost
- dabGAM ↔ loadVarRL
- dabGAM ↔ loadRL

## Example Equation (piLine to loadVarRL)

```python
def eq_piLine_loadVarRL(self, node1, node2):
    x_1, u_1, x_2, u_2 = self.get_info(node1, node2)
    Cline_r = self.graph.nodes[node1]['params'][f'R_c_{node1}']
    return sp.Matrix([
        u_1.i_out - x_2.i_L,  # Output current of piLine = inductor current of loadVarRL
        x_1.v_Cout + (x_1.i_L - u_1.i_out) * Cline_r - u_2.v_in,  # Output voltage continuity
    ])
```

## Extending

To add a new connection type, implement a new method in the class and add it to the `component_equations` dictionary. The method should return a `sympy.Matrix` of equations using the namedtuple attributes for clarity.

---

*This documentation is generated from the source code in `components/connection.py` and describes the symbolic connection logic for ODEsim power system modeling.*
