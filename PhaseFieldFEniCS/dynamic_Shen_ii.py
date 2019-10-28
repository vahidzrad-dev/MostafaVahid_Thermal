# Phase field fracture implementation in FEniCS    
# The code is distributed under a BSD license     

# If using this code for research or industrial purposes, please cite:
# Hirshikesh, S. Natarajan, R. K. Annabattula, E. Martinez-Paneda.
# Phase field modelling of crack propagation in functionally graded materials.
# Composites Part B: Engineering 169, pp. 239-248 (2019)
# doi: 10.1016/j.compositesb.2019.04.003

# Emilio Martinez-Paneda (mail@empaneda.com)
# University of Cambridge

# Preliminaries and mesh
from dolfin import *
import numpy as np
import ipdb
# from ufl import *

set_log_level(20)

# parameters of the nonlinear solver used for the d-problem
solver_d_parameters={"method", "tron", 			# when using gpcg make sure that you have a constant Hessian
               "monitor_convergence", True,
                       #"line_search", "gpcg"
                       #"line_search", "nash"
                       #"preconditioner", "ml_amg"
               "report", True}

# parameters of the nonlinear solver used for the displacement-problem
solver_u_parameters ={"linear_solver", "mumps", # prefer "superlu_dist" or "mumps" if available
            "preconditioner", "default",
            "report", False,
            "maximum_iterations", 500,
            "relative_tolerance", 1e-5,
            "symmetric", True,
            "nonlinear_solver", "newton"}

L = 50.0e-3                                     # Width: mm (Chu 2017-3.3)
H = 40.0e-3                                     # Height: mm (Chu 2017-3.3)
a = 5.0e-3                                      # 2a: mm (2a is length of crack)

# subdir = "meshes/"
# mesh_name = "mesh" # "fracking_hsize%g" % (h_size)
#
# mesh = Mesh(subdir + mesh_name + ".xml")
mesh = BoxMesh(Point(-L/2., -H/2., -a/2.), Point(L/2., H/2., a/2.), 25, 25, 5)    # Q: mesh_size?
# ipdb.set_trace()

# q_degree = 3
# dx = dx(metadata={'quadrature_degree': q_degree})

# Define Space
V_u = VectorFunctionSpace(mesh, 'CG', 1)
V_d = FunctionSpace(mesh, 'CG', 1)

u_, u, u_t = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
d_, d, d_t = Function(V_d), TrialFunction(V_d), TestFunction(V_d)


n_dim = len(u_)
# ipdb.set_trace()

# Introduce manually the material parameters
E = 116.0e9		                                # Young's modulus: (G)Pa (Chu 2017-4.1)
nu = 0.32		                                # Poisson's ratio: - (Chu 2017-4.1)
Gc = 8.6207e4		                            # critical energy release rate: J/m^2 (Chu 2017-4.1)

h_size = 1.0e-3	    	                        # mesh size: mm (Chu 2017-4.1) 'h is 10 times more than reference.'
ell = 4.0 * h_size                              # length scale: mm (Chu 2017-4.1)

lmbda = Constant(E*nu/((1+nu)*(1-2*nu)))		# Lamé constant: MPa (for plane strain)
# lmbda = Constant(E*nu/(1.0-nu**2))		    # Lamé constant: MPa (for plane stress)

mu = Constant(E/(2*(1+nu)))      				# shear modulus: MPa (conversion formulae)
cw = 1.0/2.0

rho = 4.506e3 						            # density: kg/m^3 (Chu 2017-4.1)
eta = 1.0e3
deltaT = 1.0e-3                                 # source: (Chu2017-3.3: h_size vs H?)
# print('DeltaT', deltaT)


# Constitutive functions
# strain
def epsilon_e(u_):
    return sym(grad(u_))


# strain energy
def psi(u_): # Note: The trace operator is understood in three-dimensions setting,
    # and it accommodates both plane strain and plane stress cases. Current formulation is valid for plane strain only.
    return lmbda/2.0 * (tr(epsilon_e(u_)))**2 + mu * inner(epsilon_e(u_), epsilon_e(u_))


def psi_p(u_):
    return (lmbda/2.0 + mu/3.0) * ((tr(epsilon_e(u_)) + abs(tr(epsilon_e(u_))))/2.0)**2.0 \
           + mu * inner(dev(epsilon_e(u_)), dev(epsilon_e(u_)))


def psi_n(u_):
    return (lmbda/2.0 + mu/3.0) * ((tr(epsilon_e(u_)) - abs(tr(epsilon_e(u_))))/2.0)**2.0


# stress
def sigma(u_): # Note: The trace operator is understood in three-dimensions setting,
    # and it accommodates both plane strain and plane stress cases. Current formulation is valid for plane strain only.
    return lmbda * tr(epsilon_e(u_)) * Identity(len(u_)) + 2.0 * mu * (epsilon_e(u_))


def sigma_p(u_):    # sigma_+ for Amor's model
    return (lmbda + 2.0 * mu / 3.0) * (tr(epsilon_e(u_)) + abs(tr(epsilon_e(u_))))/2.0 \
           * Identity(len(u_)) + 2.0 * mu * dev(epsilon_e(u_))


def sigma_n(u_):    # sigma_- for Amor's model
    return (lmbda + 2.0 * mu / 3.0) * (tr(epsilon_e(u_)) - abs(tr(epsilon_e(u_))))/2.0 \
           * Identity(len(u_))


# Boundary conditions #Note: We need to change manually the numbers in accordance with the mesh.
left = CompiledSubDomain("near(x[0], -25.0e-3) && on_boundary")
right = CompiledSubDomain("near(x[0], 25.0e-3) && on_boundary")
bot = CompiledSubDomain("near(x[1], -20.0e-3) && on_boundary")
top = CompiledSubDomain("near(x[1], 20.0e-3) && on_boundary")
rear = CompiledSubDomain("near(x[2], -2.5e-3) && on_boundary")
front = CompiledSubDomain("near(x[2], 2.5e-3) && on_boundary")


# ipdb.set_trace()
zero_v = Constant((0.,)*len(u_))


# Boundary conditions for v
t_cut = 1.0e-6                                     # unit: sec.
v_f = 16.5                                         # v_f is max. velocity: m/s.
Amp = 3.5e6
v_p = Expression(("0", "0", "-Amp * (1 - sin(20 * pi * t))/2"), t=0, Amp=Amp, degree=0)   \
    # velocity at part of left edge.

traction = interpolate(v_p, V_u)


# Boundary conditions for u
bc_u_left = DirichletBC(V_u, zero_v, left)
bc_u_right = DirichletBC(V_u, zero_v, right)
bc_u_bot = DirichletBC(V_u, zero_v, bot)
bc_u_top = DirichletBC(V_u, zero_v, top)


# bc_u = [DirichletBC(V_u, zero_v, left)]
bc_u = [bc_u_left, bc_u_right, bc_u_bot, bc_u_top]
# ipdb.set_trace()


# Boundary conditions for d
def crack(x):
    return (x[0]**2. + x[1]**2.) < 10.e-3 and abs(x[2] - 0.) < DOLFIN_EPS


bc_d = [DirichletBC(V_d, Constant(0.0), rear)]


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top.mark(boundaries,1)
bot.mark(boundaries,2)
left.mark(boundaries,3)
right.mark(boundaries,4)
rear.mark(boundaries,5)
front.mark(boundaries,6)

ds = Measure("ds")(subdomain_data=boundaries)
n = FacetNormal(mesh)


# u0 = Function(V_u)
# v0 = Function(V_u)
# a0 = Function(V_u)

# u0.assign(zero_v)
# v0.assign(zero_v)
# a0.assign(zero_v)

# ipdb.set_trace()


# update v
# def update_a(u_, u0, v0, a0, ufl=True):
#     return 4.0/deltaT * (u_ - u0 - v0 * deltaT - deltaT**2.0/4.0 * a0)


def update_u(u_old, v_old, a_old, ufl=True):
    return u_old + deltaT * v_old + (deltaT**2)/2.0 * a_old


def update_v(v_old, a_old, ufl=True):
    return v_old + deltaT/2.0 * a_old


# def update_v_ii(v_old, a_old, a_new, ufl=True):
#     return update_v(v_old, a_old, ufl=True) + deltaT/2.0 * a_new


def update_d_i(d_old, r_old, ufl=True):
    return d_old + deltaT * r_old


def update_d_ii(d_old, r_old, ufl=True):
    return d_old + deltaT * (r_old + r_new)/2.


u0 = interpolate(zero_v, V_u)
v0 = interpolate(zero_v, V_u)
a0 = interpolate(zero_v, V_u)
d0 = interpolate(Constant(0.0), V_d)


# def update_fields(u_, u0, v0, a0, ufl=True):
#     """Update fields at the end of each time step."""
#
#     # Get vectors (references)
#     u_vec, u0_vec = u_.vector(), u0.vector()
#     v0_vec, a0_vec = v0.vector(), a0.vector()
#
#     # use update functions using vector arguments
#     a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=True)
#     v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=True)
#
#     # Update (u_old <- u)
#     v0.vector()[:], a0.vector()[:] = v_vec, a_vec
#     # u0.vector()[:] = u_.vector()
    
# ipdb.set_trace()


# weak forms
E_u = (1.0 - d_)**2.0 * psi_p(u_) * dx + psi_n(u_) * dx
E_d = 1.0/(4.0 * cw) * Gc * (d_**2.0/ell * dx + ell * inner(grad(d_), grad(d_)) * dx)

Pi = E_u + E_d

Du_Pi = derivative(Pi, u_, u_t)
J_u = derivative(Du_Pi, u_, u)

# J_m = PETScMatrix()
# J_k = PETScVector()


class UProblem(OptimisationProblem):
    
    def __init__(self):
        OptimisationProblem.__init__(self)
    
    # Objective function
    def f(self, x):
        u_.vector()[:] = x
        return assemble(Pi)
    
    # Gradient of the objective function
    def F(self, b, x):
        u_.vector()[:] = x
        assemble(Du_Pi, tensor=b)
    
    # Hessian of the objective function
    def J(self, A, x):
        u_.vector()[:] = x
        assemble(J_u, tensor=A)


problem_u = UProblem()

# Mass form
def mass_u(u, u_t):
    return rho*inner(u, u_t) * dx


# Elastic stiffness form
def k_u(u, u_t, d_):
    return inner((1.-d_)**2. * sigma(u), grad(u_t)) * dx


def force_u(u_t):
    return dot(traction, u_t) * ds(6)


# class DamageProblem(OptimisationProblem):
#
#     def __init__(self):
#         OptimisationProblem.__init__(self)
#
#     # Objective function
#     def f(self, x):
#         d_.vector()[:] = x
#         return assemble(Pi)
#
#     # Gradient of the objective function
#     def F(self, b, x):
#         d_.vector()[:] = x
#         assemble(Dd_Pi, tensor=b)
#
#     # Hessian of the objective function
#     def J(self, A, x):
#         d_.vector()[:] = x
#         assemble(J_d, tensor=A)

# mass_u_matrix = assemble(mass_u(u, u_t))
# k_u_matrix = assemble(k_u(u, u_t, d0))
# force_u_matrix = assemble(force_u(u_t))

# mass_form = rho*inner(u, u_t)
# mass_action_form = action(mass_form, Constant(1.0))
M_m = PETScMatrix()
assemble(mass_u(u, u_t), tensor=M_m)

M_k = PETScMatrix()
assemble(k_u(u, u_t, d0), tensor=M_k)

M_f = PETScVector()
assemble(force_u(u_t), tensor=M_f)
f_vec = M_f.vec()

# M_lumped.zero()
# M_lumped.set_diagonal(assemble(mass_action_form))
m_mat = M_m.mat()
inv_m = m_mat.convert('dense')

k_mat = M_k.mat()
inv_k = k_mat.convert()
KM = inv_m.matMult(k_mat)

# J_mat = J_u.mat()
# MF = inv_m.matMult(J_mat)

# KM * u_.vector()[:]
# inv_m * M_k
# ipdb.set_trace()

# A = inner(k_u_matrix, force_u_matrix)
# B = inner(mass_u_matrix, k_u_matrix)
# C = inner(mass_u_matrix, mass_u_matrix)
# print(C)

# res_u = mass_u_matrix + k_u_matrix - force_u_xmatrix
# # # Vahid: See how to handle non-symmetric cases
# a_form = lhs(res_u)
# L_form = rhs(res_u)
#
# K, res_u = assemble_system(a_form, L_form, bc_u)
# solver = LUSolver(K, "mumps")
# solver.parameters["symmetric"] = True


# ipdb.set_trace()


def mass_d(d, d_t):
    return Gc/ell * inner(d, d_t) * dx


def k_d(d, d_t):
    return Gc * ell**2. * dot(nabla_grad(d), grad(d_t)) * dx


def force_d(u_, d_t):
    return -2. * psi_p(u_) * d_t * dx


def Y_d(u_, d, d_t):
    return 2. * psi_p(u_) * d * d_t * dx - Gc/ell * d * d_t * dx \
           - Gc * ell**2. * inner(grad(d), grad(d_t)) * dx


# mass_d_matrix = assemble(mass_d(d, d_t))
# k_d_matrix = assemble(k_d(d, d_t))
force_d_matrix = assemble(force_d(u_, d_t))
Y_d_matrix = assemble(Y_d(u_, d, d_t))
ipdb.set_trace()


# Initialization of the iterative procedure and output requests
min_step = 0
max_step = 0.2
n_step = 201
load_multipliers = np.linspace(min_step, max_step, n_step)
max_iterations = 100

tol = 1e-3
conc_u = File("./ResultsDir_Shen/u.pvd")
conc_d = File("./ResultsDir_Shen/d.pvd")

# Staggered scheme
for (i_p, p) in enumerate(load_multipliers):

    itr = 0
    err = 1.0

    v_p.t = p
    print(p)
    # ipdb.set_trace()

    # res_u = assemble(L_form)
    # bc_u.apply(res_u)
    # solver.solve(K, u.vector(), res_u)

    ipdb.set_trace()

# fname.close()
print ('Simulation completed')