#%%
%reload_ext autoreload
%autoreload 2
#%%
from tests.piLine import Test_piLine

# Create an instance of the test class
test = Test_piLine(methodName='test_parameter_grid')
test.setUpClass()

# Run the test directly
try:
    test.test_parameter_grid()
finally:
    test.tearDownClass()

#%%
from tests.c_VsourceR_piLine import *

# Create an instance of the test class
test = Test_VsourceR_piLine(methodName='test_parameter_grid')
test.setUpClass()

# Run the test directly
try:
    test.test_parameter_grid()
finally:
    test.tearDownClass()

#%%

from tests.delayPI import *

# Create an instance of the test class
test = Test_delayPI(methodName='test_parameter_grid')
test.setUpClass()

# Run the test directly
try:
    test.test_parameter_grid()
finally:
    test.tearDownClass()

#%%
from tests.loadVarRL import *

# Create an instance of the test class
test = Test_loadVarRL(methodName='test_parameter_grid')
test.setUpClass()

# Run the test directly
try:
    test.test_parameter_grid()
finally:
    test.tearDownClass()

#%%
from tests.loadRL import *

# Create an instance of the test class
test = Test_loadRL(methodName='test_parameter_grid')
test.setUpClass()

# Run the test directly
try:
    test.test_parameter_grid()
finally:
    test.tearDownClass()

#%%
from tests.delayLpPIDroop import *

# Create an instance of the test class
test = Test_delayLpPIDroop(methodName='test_parameter_grid')
test.setUpClass()

# Run the test directly
try:
    test.test_parameter_grid()
finally:
    test.tearDownClass()

#%%
from tests.dabGAM import *

# Create an instance of the test class
test = Test_dabGAM(methodName='test_parameter_grid')
test.setUpClass()

# Run the test directly
try:
    test.test_parameter_grid()
finally:
    test.tearDownClass()

#%%
from tests.c_KCL import *

# Create an instance of the test class
test = Test_KCL(methodName='test_parameter_grid')
test.setUpClass()

# Run the test directly
try:
    test.test_parameter_grid()
finally:
    test.tearDownClass()
#%%
test.grid.x
#%%
test.grid.plot_states(states = ['v_Cin_1'])
test.grid.plot_states(states = ['v_Cout_1'])
