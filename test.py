from firedrake import *
from ufl.algorithms.apply_integral_scaling import apply_integral_scaling
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
from ufl.algorithms.apply_derivatives import apply_derivatives
from transform import pull_back_form

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)
Phi = Function(W)
u = TrialFunction(V)
v = TestFunction(V)
eqn = u*v*dx + u*v*dS
eqn = apply_integral_scaling(eqn)
eqn = apply_geometry_lowering(eqn)
eqn = pull_back_form(eqn, Phi)
