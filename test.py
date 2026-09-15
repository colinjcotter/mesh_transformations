from firedrake import *
from ufl.algorithms.apply_integral_scaling import apply_integral_scaling
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
from ufl.algorithms.apply_derivatives import apply_derivatives
from transform import pull_back_form

PETSc.Sys.popErrorHandler()

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
Phi = Function(W)
u = TrialFunction(V)
v = TestFunction(V)
eqn = (u*v + inner(grad(u), grad(v)))*dx(degree=2)
fcp = {"do_apply_integral_scaling": False}
F = x*v*dx(degree=2)
eqn = apply_function_pullbacks(eqn)
eqn = apply_integral_scaling(eqn)
eqn = pull_back_form(eqn, Phi)
print(F)
#F = apply_function_pullbacks(F)
print(F)
F = apply_integral_scaling(F)
F = pull_back_form(F, Phi)

u0 = Function(V)
solve(eqn == F, u0, form_compiler_parameters=fcp)
