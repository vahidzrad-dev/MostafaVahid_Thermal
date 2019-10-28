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
H = 9.8e-3                                      # Height: mm (Chu 2017-3.3)
# a = 5.0e-3                                    # 2a: mm (2a is length of crack)

subdir = "meshes/"
mesh_name = "mesh" # "fracking_hsize%g" % (h_size)

mesh = Mesh(subdir + mesh_name + ".xml")

# Define Space
V_u = VectorFunctionSpace(mesh, 'CG', 1)
V_d = FunctionSpace(mesh, 'CG', 1)
V_T = FunctionSpace(mesh, 'CG', 1)

u_, u, u_t = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
d_, d, d_t = Function(V_d), TrialFunction(V_d), TestFunction(V_d)
T_, T, T_t = Function(V_T), TrialFunction(V_T), TestFunction(V_T)
# H_ = Function(V_d)

n_dim = len(u_)
# Introduce manually the material parameters
E = 380.0e9		                                # Young's modulus: MPa (Chu 2017-4.1)
nu = 0.25		                                # Poisson's ratio: - (Chu 2017-4.1)
Gc = 26.95		                                # critical energy release rate: MPa-mm (Chu 2017-4.1)

h_size = 5.0e-4		                            # mesh size: mm (Chu 2017-4.1)
ell = 4.0 * h_size                              # length scale: mm (Chu 2017-4.1)

Ts = Constant(680.0)  	                        # initial temperature of slab: K (Chu 2017-3.3)
Tw = Constant(300.0)                            # temperature of surface contacted with water: K (Chu 2017-3.3)

lmbda = Constant(E*nu/((1+nu)*(1-2*nu)))		# Lamé constant: MPa (for plane strain)
# lmbda = Constant(E*nu/(1.0-nu**2))		    # Lamé constant: MPa (for plane stress)

mu = Constant(E/(2*(1+nu)))      				# shear modulus: MPa (conversion formulae)

rho = 3.9e3 						            # density: kg/m^3 (Chu 2017-4.1)

alpha = Constant(6.6e-6)                        # linear expansion coefficient: 1/K (Chu 2017-4.1)

c = Constant(961.5) 					        # specific heat of material: #J/(kgK) (Chu 2017-4.1)
k = Constant(21.0)  					        # thermal conductivity: #W/(mK)=J/(mKs) (Chu 2017-4.1)
cw = 2.0/3.0                                    # To choose if AT1 or AT2 model is used.

deltaT = rho * c * h_size**2 / k                 # source: (Chu2017-3.3: h_size vs H?)
# print('DeltaT', deltaT)

# Constitutive functions
# strain


def epsilon(u_):
    return sym(grad(u_))


def epsilon_t(T_):
    return alpha * T_ * Identity(n_dim)


def epsilon_e(u_, T_):
    return epsilon(u_) - epsilon_t(T_)


# def epsilon_e(u_, T_):
#     return sym(grad(u_))


# strain energy
def psi(u_, T_):
    return lmbda/2.0 * tr(epsilon_e(u_, T_))**2.0 + mu * inner(epsilon_e(u_, T_), epsilon_e(u_, T_))


def psi_n(u_, T_):
    return psi(u_, T_) - 0.5 * (lmbda + mu) * tr(epsilon_e(u_, T_))**2.0


def psi_p(u_, T_):      # This shall be used everywhere.
    return psi(u_, T_) * (tr(epsilon_e(u_, T_)) + abs(tr(epsilon_e(u_, T_))))/2.0 \
           + psi_n(u_, T_) * (tr(epsilon_e(u_, T_)) - abs(tr(epsilon_e(u_, T_))))/2.0


# stress
def sigma(u_, T_):      # no decomposition
    return derivative(psi(u_, T_), epsilon(u_))


def sigma_p(u_, T_):    # sigma_+ for Amor's model
    return (lmbda + 2.0 * mu / 3.0) * (tr(epsilon_e(u_, T_)) + abs(tr(epsilon_e(u_, T_))))/2.0 * Identity(len(u_)) \
           + 2.0 * mu * dev(epsilon_e(u_, T_))


def sigma_n(u_, T_):    # sigma_- for Amor's model
    return (lmbda + 2.0 * mu / 3.0) * (tr(epsilon_e(u_, T_)) - abs(tr(epsilon_e(u_, T_))))/2.0 * Identity(len(u_))


# # stress
# def sigma(u_, T_):      # no decomposition
#     return lmbda * tr(epsilon_e(u_, T_)) * Identity(len(u_)) + 2.0 * mu * (epsilon_e(u_, T_))
#
#
# def sigma_p(u_, T_):    # sigma_+ for Amor's model
#     return (lmbda + 2.0 * mu / 3.0) * (tr(epsilon_e(u_, T_)) + abs(tr(epsilon_e(u_, T_))))/2.0 * Identity(len(u_)) \
#            + 2.0 * mu * dev(epsilon_e(u_, T_))
#
#
# def sigma_n(u_, T_):    # sigma_- for Amor's model
#     return (lmbda + 2.0 * mu / 3.0) * (tr(epsilon_e(u_, T_)) - abs(tr(epsilon_e(u_, T_))))/2.0 * Identity(len(u_))


# def psi_p(u_, T_):
#     return (lmbda/2.0 + mu/3.0) * ((tr(epsilon_e(u_, T_)) + abs(tr(epsilon_e(u_, T_))))/2.0)**2.0 \
#            + mu * inner(dev(epsilon_e(u_, T_)), dev(epsilon_e(u_, T_)))


# def psi_n(u_, T_):
#     return (lmbda/2.0 + mu/3.0) * ((tr(epsilon_e(u_, T_)) - abs(tr(epsilon_e(u_, T_))))/2.0)**2.0

# def H(u_, T_):
#     return 0.5 * (abs(psi_n(u_, T_) - psi_p(u_, T_)) + abs(psi_n(u_, T_) + psi_p(u_, T_)))


# Boundary conditions
top = CompiledSubDomain("near(x[1], 9.8e-3) && on_boundary")
bot = CompiledSubDomain("near(x[1], 0.0) && on_boundary")
left = CompiledSubDomain("near(x[0], 0.0) && on_boundary")
right = CompiledSubDomain("near(x[0], 50.0e-3) && on_boundary")


class Pinpoint(SubDomain):
    TOL = 1e-3
    def __init__(self, coords):
        self.coords = np.array(coords)
        SubDomain.__init__(self)
    def move(self, coords):
        self.coords[:] = np.array(coords)
    def inside(self, x, on_boundary):
        TOL = 1e-3
        return np.linalg.norm(x-self.coords) < TOL


pinpoint_l = Pinpoint([0.,0.])
pinpoint_r = Pinpoint([L,0.])

load_top = Expression(("0.0", "t"), t=0.0, degree=1)
load_bot = Expression(("0.0", "0.0"), t=0.0, degree=1)
# load_bot = Expression(("0.0", "0.0"))

# Boundary conditions for u
bc_u_pt_left = DirichletBC(V_u.sub(1), Constant(0.0), pinpoint_l, method='pointwise')
bc_u_pt_right = DirichletBC(V_u.sub(1), Constant(0.0), pinpoint_r, method='pointwise')

bc_u_bot = DirichletBC(V_u, load_bot, bot)
bc_u_top = DirichletBC(V_u, load_top, top)
bc_u_right = DirichletBC(V_u, Constant([0.0, 0.0]), right)
bc_u_left = DirichletBC(V_u, Constant([0.0, 0.0]), left)
bc_u = [bc_u_right, bc_u_pt_right, bc_u_pt_left]

# def Crack(x):
#     return abs(x[1] - H/2.0) < DOLFIN_EPS and abs(x[0] - L/2.0) <= a

# bc_d_left = DirichletBC(V_d, Constant(0.0), left)
bc_d_right = DirichletBC(V_d, Constant(0.0), right)
# bc_d = [DirichletBC(V_d, Constant(0.0), right)]
bc_d = [bc_d_right]

# Boundary conditions for T
bc_T_top = DirichletBC(V_T, Tw, top)
bc_T_bot = DirichletBC(V_T, Tw, bot)
bc_T_left = DirichletBC(V_T, Tw, left)
bc_T = [bc_T_top, bc_T_bot, bc_T_left]

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top.mark(boundaries,1)
bot.mark(boundaries,2)
left.mark(boundaries,3)
right.mark(boundaries,4)
ds = Measure("ds")(subdomain_data=boundaries)
n = FacetNormal(mesh)

zero_v = Constant((0.,)*2)
u0 = interpolate(zero_v, V_u)

d0 = interpolate(Constant(0.0), V_d)
T0 = interpolate(Expression('T_init', T_init = Ts, degree=1), V_T)

# Energy form
E_u = (1.0 - d_)**2.0 * psi_p(u_, T_) * dx + psi_n(u_, T_) * dx
E_d = 1.0/(4.0 * cw) * Gc * (d_/ell * dx + ell * inner(grad(d_), grad(d_)) * dx)
# E_T = (1.0 - d_)**2.0 * rho * c * (T_ - T0) * T * dx - deltaT * (1.0 - d_)**2 * k * inner(grad(T_), grad(T)) * dx

# weak form with refer to Chu 2017
# E_T = (1.0 - d_)**2 * rho * c * (T_ - T0) * T * dx - deltaT * (1.0 - d_)**2 * k * inner(grad(T_), grad(T)) * dx

# E_u = (1.0 - d_)**2.0 * inner(sigma_p(u_, T_), epsilon_e(u_t, T_t)) * dx + \
#       (1.0 - d_)**2.0 * inner(sigma_p(u_, T_), epsilon_e(u_t, T_t)) * dx + \
#       (1.0 - d_)**2.0 * inner(dev(sigma(u_, T_)), dev(u_t, T_t)) * dx
#
# E_d = -2.0 * (1.0 - d_) * d_t * inner(dev(sigma(u_, T_)), dev(u_, T_)) * dx + \
#     + Gc/(4.0 * cw)/ell * (2.0 * d_ * d_t + 2.0 * ell**2.0 * inner(grad(d_), grad(d_t))) * dx

E_T = rho * c / deltaT * (T_ - T0) * T_t * dx - k * inner(grad(T_), grad(T_t)) * dx
# End

Pi = E_u + E_d

Du_Pi = derivative(Pi, u_, u_t)
J_u = derivative(Du_Pi, u_, u)
problem_u = NonlinearVariationalProblem(Du_Pi, u_, bc_u, J_u)
solver_u = NonlinearVariationalSolver(problem_u)
prm = solver_u.parameters
prm["newton_solver"]["absolute_tolerance"] = 1E-8
prm["newton_solver"]["relative_tolerance"] = 1E-7
prm["newton_solver"]["maximum_iterations"] = 25
prm["newton_solver"]["relaxation_parameter"] = 1.0
prm["newton_solver"]["preconditioner"] = "default"
prm["newton_solver"]["linear_solver"] = "mumps"

Dd_Pi = derivative(Pi, d_, d_t)
J_d = derivative(Dd_Pi, d_, d)


class DamageProblem(OptimisationProblem):
    
    def __init__(self):
        OptimisationProblem.__init__(self)
    
    # Objective function
    def f(self, x):
        d_.vector()[:] = x
        return assemble(Pi)
    
    # Gradient of the objective function
    def F(self, b, x):
        d_.vector()[:] = x
        assemble(Dd_Pi, tensor=b)
    
    # Hessian of the objective function
    def J(self, A, x):
        d_.vector()[:] = x
        assemble(J_d, tensor=A)


# Create the PETScTAOSolver
problem_d = DamageProblem()

# Parse (PETSc) parameters
parameters.parse()

solver_d = PETScTAOSolver()

d_lb = interpolate(Expression("0.", degree=1), V_d)  # lower bound, set to 0
d_ub = interpolate(Expression("1.", degree=1), V_d)  # upper bound, set to 1

for bc in bc_d:
    bc.apply(d_lb.vector())

for bc in bc_d:
    bc.apply(d_ub.vector())

J_T = derivative(E_T, T_, T)
problem_T = NonlinearVariationalProblem(E_T, T_, bc_T, J=J_T)
solver_T = NonlinearVariationalSolver(problem_T)
# prmT = solver_T.parameters
# prmT["newton_solver"]["absolute_tolerance"] = 1E-8
# prmT["newton_solver"]["relative_tolerance"] = 1E-7
# prmT["newton_solver"]["maximum_iterations"] = 25
# prmT["newton_solver"]["relaxation_parameter"] = 1.0
# prmT["newton_solver"]["preconditioner"] = "default"
# prmT["newton_solver"]["linear_solver"] = "mumps"

# Initialization of the iterative procedure and output requests
min_step = 0
max_step = 1.0
n_step = 101
load_multipliers = np.linspace(min_step, max_step, n_step)
max_iterations = 100

tol = 1e-3
conc_u = File ("./ResultsDir/u.pvd")
conc_d = File ("./ResultsDir/d.pvd")
conc_T = File ("./ResultsDir/T.pvd")

# fname = open('ForcevsDisp.txt', 'w')

# u_.vector()[:] = u0.vector()
solver_u.solve()
u0.vector()[:] = u_.vector()

# d_.vector()[:] = d0.vector()
solver_d.solve(problem_d, d_.vector(), d_lb.vector(), d_ub.vector())
d0.vector()[:] = d_.vector()

solver_T.solve()
T0.vector()[:] = T_.vector()
# ipdb.set_trace()

# Staggered scheme
for (i_p, p) in enumerate(load_multipliers):

    itr = 0
    err = 1.0
    # load_top.t = 1.0e-7 * p
    # load_bot.t = 1.0e-5 * p
    # print('Load: ', load_top.t)
    
    while err > tol and itr < max_iterations:
        itr += 1
        solver_u.solve()
        solver_d.solve(problem_d, d_.vector(), d_lb.vector(), d_ub.vector())
        # solver_T.solve()
    #
        err_u = errornorm(u_, u0, norm_type = 'l2', mesh = None)
        err_d = errornorm(d_, d0, norm_type = 'l2', mesh = None)
        # err_T = errornorm(T_, T0, norm_type = 'l2', mesh = None)
    #
        err = max(err_d, err_u)
        # err = err_d
        # print('err_u: ', err_u)
        
        u0.vector()[:] = u_.vector()
        d0.vector()[:] = d_.vector()
    
        if err < tol:
            print ('Iterations:', itr, ', Total time', p)
            conc_d << d_
            conc_u << u_
    #
    #     # Traction = dot(sigma(u_, T_), n)
    #     # fy = Traction[1]*ds(1)
    #
    #     # fname.write(str(p*u_r) + "\t")
    #     # fname.write(str(assemble(fy)) + "\n")
    
    itr = 0
    err = 1.0
    while err > tol and itr < max_iterations:
        itr += 1
        solver_T.solve()
        err_T = errornorm(T_, T0, norm_type = 'l2', mesh = None)
    # #
    # #     # # # T0.assign(T_)
        T0.vector()[:] = T_.vector()
    # #     if err < tol:
    # #         print ('Iterations:', itr, ', Total time', p)
    conc_T << T_


# fname.close()
print ('Simulation completed')