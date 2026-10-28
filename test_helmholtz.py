from firedrake import *
from transform import pullback

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

Phi = Function(W)
Phi0 = Function(W)

u = TrialFunction(V)
v = TestFunction(V)
x, y = SpatialCoordinate(mesh)
Phi.interpolate(as_vector([x+y**2,y]))
mesh.coordinates.assign(Phi)

Phi.interpolate(as_vector([x,y+(x-0.5)**2]))
Phi0.interpolate(as_vector([x,y]))

# solution using our pullback
#bcs=[]
bcs=[DirichletBC(V, Constant(0.), "on_boundary")]
#eqn0 = u*v*dx(degree=2)
eqn0 = (inner(grad(u), grad(v)))*dx(degree=2)
q0 = Function(V,name="q0").interpolate(Phi[0]*Phi[1])
F = q0*v*dx(degree=2)

eqn = pullback(eqn0, Phi)
F = pullback(F, Phi)

u0 = Function(V, name="on reference")
fcp = {"do_apply_integral_scaling": False,
       "do_apply_function_pullbacks": False}
solve(eqn == F, u0, bcs=bcs, form_compiler_parameters=fcp)

# solution where the mesh has actually moved
mesh.coordinates.assign(Phi)

q = Function(V,name="q").interpolate(x*y)
F = q*v*dx(degree=2)
u1 = Function(V, name="on deformed mesh")
solve(eqn0 == F, u1, bcs=bcs)

# move the mesh back
VTKFile("moved.pvd").write(u1,q0)
mesh.coordinates.assign(Phi0)

VTKFile("reference.pvd").write(u0, u1, q0, q)

print(norm(u0-u1))
