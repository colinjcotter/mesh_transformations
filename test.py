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
Phi = Function(W)
u = TrialFunction(V)
v = TestFunction(V)
eqn = (u*v + inner(grad(u), grad(v)))*dx(degree=2)
fcp = {"do_apply_integral_scaling": False,
       "do_apply_function_pullbacks": False}
x, y = SpatialCoordinate(mesh)
q = Function(V) # .interpolate(1)
F = q*v*dx(degree=2)
eqn = apply_function_pullbacks(eqn)
eqn = apply_integral_scaling(eqn)
eqn = apply_derivatives(eqn)
eqn = apply_geometry_lowering(eqn)
eqn = pull_back_form(eqn, Phi)
F = apply_function_pullbacks(F)
F = apply_integral_scaling(F)
F = apply_derivatives(F)
F = apply_geometry_lowering(F)
F = pull_back_form(F, Phi)

print(eqn)
print(F)

u0 = Function(V)
solve(eqn == F, u0, form_compiler_parameters=fcp)
