from firedrake import *
from ufl.algorithms.apply_integral_scaling import apply_integral_scaling
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
from ufl.algorithms.apply_derivatives import apply_derivatives
from transform import MeshTransformPullbacks

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)
Phi = Function(W)
u = TrialFunction(V)
v = TestFunction(V)
eqn = u*v*dx
eqn = apply_integral_scaling(eqn)
eqn = apply_geometry_lowering(eqn)

MTP = MeshTransformPullbacks(Phi)
integrand1 = eqn.integrals()[0].integrand()
integrand2 = MTP(integrand1)
