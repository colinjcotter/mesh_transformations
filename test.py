from firedrake import *
from transform import pullback

PETSc.Sys.popErrorHandler()

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)
Phi = Function(W)
u = TrialFunction(V)
v = TestFunction(V)
eqn = (u*v + inner(grad(u), grad(v)))*dx(degree=2)
x, y = SpatialCoordinate(mesh)
q = Function(V) # .interpolate(1)
F = q*v*dx(degree=2)

eqn = pullback(eqn, Phi)
F = pullback(F, Phi)

u0 = Function(V)
fcp = {"do_apply_integral_scaling": False,
       "do_apply_function_pullbacks": False}
solve(eqn == F, u0, form_compiler_parameters=fcp)
