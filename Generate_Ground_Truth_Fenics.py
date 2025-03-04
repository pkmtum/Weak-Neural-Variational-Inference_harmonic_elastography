import numpy as np
import fenics as fe
import torch
import matplotlib.pyplot as plt

def bottom(x, on_boundary):
    return (on_boundary and fe.near(x[1], 0.0))

def left(x, on_boundary):
    return (on_boundary and fe.near(x[0], 0.0))

def top(x, on_boundary):
    return (on_boundary and fe.near(x[1], 10.0))

def right(x, on_boundary):
    return (on_boundary and fe.near(x[0], 10.0))


# Strain function
def epsilon(u):
    return 0.5*(fe.grad(u) + fe.grad(u).T)


# Stress function
def sigma(u, lmbda, mu):
    return lmbda*fe.div(u)*fe.Identity(2) + 2*mu*epsilon(u)

# %%%%%%%%%%%%%%%%% GEGEBEN %%%%%%%%%%%%%%%%%
# Neumann
rhs = {}
rhs["neumann_right"] = np.array([0.0, 0.0])
rhs["neumann_top"] = np.array([0.0, -0.15])

# Body force
f = np.array([0.0, 0.0])

# Material parameters
nu = 0.3
E_out = 10.0
E_in = 50.
rho = 1e-6
omega = 200 * 3.14 * 2

# %%%%%%%%%%%%%%%% PROGRAMM %%%%%%%%%%%%%%%%%
# Define the mesh
mesh_dim = 64
mesh = fe.RectangleMesh(fe.Point(0.0, 0.0), fe.Point(10.0, 10.0), mesh_dim, mesh_dim)
x=fe.SpatialCoordinate(mesh)

tol = 1E-14
E_expr = fe.Expression('pow((x[1]-5),2) + pow((x[0]-5),2) <= 2*2 + tol ? k_0 : k_1', degree=0, tol=tol, k_0=E_in, k_1=E_out)
# flag_nonlin_expr = fe.Expression('pow((x[1]-0.5),2) + pow((x[0]-0.5),2) <= 0.25*0.25 + tol ? 1 : 0', degree=0, tol=tol)

# subdomains = fe.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
# for cell in fe.cells(mesh):
#     M = cell.midpoint()
#     if M.y()* M.y() + M.x() * M.x() <= 0.25*0.25:
#         subdomains[cell] = 1

# pl1 = fe.plot(subdomains)
# plt.colorbar(pl1)
# plt.show()

# dx = fe.Measure('dx', domain=mesh, subdomain_data=subdomains)

# Define the function space
V = fe.VectorFunctionSpace(mesh, 'P', 1)
W = fe.FunctionSpace(mesh, 'DG', 0)

# Define the trial and test functions
u = fe.Function(V)
v = fe.TestFunction(V)

# Define the conductivity field as a FEniCS function
c = fe.Function(W)
E = fe.interpolate(E_expr, W)
# this toggles the nonlinear part of the constitutive law
# flag_nonlin = fe.interpolate(flag_nonlin_expr, W)

pl2 = fe.plot(E)
plt.colorbar(pl2)
plt.show()


# --------------------
# Material fields
# --------------------
mu = E / (2*(1 + nu))
lmbda = E*nu / ((1 + nu)*(1 - 2*nu))

# --------------------
# Boundary conditions (Dirichlet)
# --------------------
bc1 = fe.DirichletBC(V.sub(1), fe.Constant(0.0), bottom)
bc2 = fe.DirichletBC(V.sub(0), fe.Constant(0.0), bottom)
bcs = [bc1, bc2]

# --------------------
# Boundary conditions (Neumann)
# --------------------
# create a mesh function which assigns an unsigned integer (size_t) to each edge
mf = fe.MeshFunction("size_t", mesh, 1) # 3rd argument is dimension of an edge
mf.set_all(0) # initialize the function to zero
# create a SubDomain subclass, specifying the portion of the boundary with x[0] < 1/2
class Boundary_right(fe.SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and fe.near(x[0], 10.0))
class Boundary_top(fe.SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and fe.near(x[1], 10.0))
B_right = Boundary_right() # instantiate it
B_top = Boundary_top() # instantiate it
# use this lefthalf object to set values of the mesh function to 1 in the subdomain
B_right.mark(mf, 1)
B_top.mark(mf, 2)
# define a new measure ds based on this mesh function
ds = fe.Measure('ds', domain=mesh, subdomain_data=mf)

r1, r2 = rhs["neumann_right"][0], rhs["neumann_right"][1]
t1, t2 = rhs["neumann_top"][0], rhs["neumann_top"][1]
r1 = fe.Constant(r1)
r2 = fe.Constant(r2)
t1 = fe.Constant(t1)
t2 = fe.Constant(t2)
g_right = fe.Expression(('r1', 'r2'), r1=r1, r2=r2, degree=1)
g_top = fe.Expression(('t1', 't2'), t1=t1, t2=t2, degree=1)
# g = fe.Expression(('x[0] > 1.0-1e-14 ? rhs : 0', 'x[1] > 1.0-1e-14 ? rhs : 0'), rhs=rhs, degree=1)

f1 = fe.Constant(f[0])
f2 = fe.Constant(f[1])
f = fe.Expression(('f1', 'f2'), f1=f1, f2=f2, degree=1)

# --------------------
# Weak form
# --------------------
# Solve the PDE
# u = fe.Function(V)

a = fe.inner(sigma(u, lmbda, mu), epsilon(v))*fe.dx - rho * omega**2 * fe.inner(u, v) * fe.dx # (0) + fe.inner(sigma_incl(u, lmbda, mu), epsilon(v))*dx(1)
# a = fe.inner(sigma(u, lmbda, mu), epsilon(v))*fe.dx
# L = fe.inner(g, v)*fe.ds # fe.dot(fe.Constant(rhs), v)*fe.dx -
L = fe.inner(g_top, v)*ds(2) #+ fe.inner(f, v)*fe.dx
F = a - L

fe.solve(F == 0, u, bcs)
#fe.plot(u)

fe.plot(u, mode="displacement")
plt.show()
fgr = fe.plot(u.sub(0))
plt.colorbar(fgr)
plt.show()
fgr = fe.plot(u.sub(1))
plt.colorbar(fgr)
plt.show()
fe.plot(E)
plt.show()

u_array = u.compute_vertex_values(mesh)

# # Reshape the numpy array to a 2D grid
u_grid = np.reshape(u_array, (2, mesh_dim+1, mesh_dim+1))

u_torch = torch.from_numpy(u_grid)

# return u_torch