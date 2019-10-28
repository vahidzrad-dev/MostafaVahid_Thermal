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

q_degree = 3
dx = dx(metadata={'quadrature_degree': q_degree})

# Define Space
V_u = VectorFunctionSpace(mesh, 'CG', 1)
# V_v = VectorFunctionSpace(mesh, 'CG', 1)
V_s = TensorFunctionSpace(mesh, 'DG', 0)
V_d = FunctionSpace(mesh, 'CG', 1)

u_, u, u_t = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
# v_, v, v_t = Function(V_v), TrialFunction(V_v), TestFunction(V_v)
v_, v, v_t = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
# s_ = Function(V_s)
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

# def psi(u_, T_): #Vahid: There is an alternative for decomposition of tension/compression in Marigo
#     # if tr(epsilon_e(u_, T_)) >= 0.:
#     if tr(epsilon_e(u_, T_)) >= uu0:
#         return psi_p(u_, T_)
#     else:
#         return psi_n(u_, T_)


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

# class v_D(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and x[1]<=a and near(x[0], 0.0)
#
# v_D = v_D()

# class Pinpoint(SubDomain):
#     TOL = 1e-3
#     def __init__(self, coords):
#         self.coords = np.array(coords)
#         SubDomain.__init__(self)
#     def move(self, coords):
#         self.coords[:] = np.array(coords)
#     def inside(self, x, on_boundary):
#         TOL = 1e-3
#         return np.linalg.norm(x-self.coords) < TOL
#
#
# pinpoint_l = Pinpoint([0.,0.])
# pinpoint_r = Pinpoint([L,0.])

# Boundary conditions for v
t_cut = 1.0e-6                                     # unit: sec.
v_f = 16.5                                         # v_f is max. velocity: m/s.
# v_R = Expression(("t <= tc ? v_f*t/tc : v_f", "0"), t=0, tc=t_cut, v_f=v_f, degree=0)\
# velocity at part of left edge.
Amp = 3.5e6
v_p = Expression(("0", "0", "-Amp * (1 - sin(20 * pi * t))/2"), t=0, Amp=Amp, degree=0)   \
    # velocity at part of left edge.

traction = interpolate(v_p, V_u)

# v_lb = interpolate(v_R, V_u)
# v0.assign(update_v(u_, u0, v0, a0))

# Boundary conditions for u
bc_u_left = DirichletBC(V_u, zero_v, left)
bc_u_right = DirichletBC(V_u, zero_v, right)
bc_u_bot = DirichletBC(V_u, zero_v, bot)
bc_u_top = DirichletBC(V_u, zero_v, top)

bc_u = [bc_u_left, bc_u_right, bc_u_bot, bc_u_top]
# ipdb.set_trace()


# Boundary conditions for v
# def v_D(x):
#     return on_boundary and abs(x[0]) < DOLFIN_EPS and x[1] <= a


# Boundary conditions for d
def crack(x):
    return (x[0]**2. + x[1]**2.) < 10.e-3 and abs(x[2] - 0.) < DOLFIN_EPS


# bc_d = [DirichletBC(V_d, Constant(1.0), crack)]
# bc_d_left = DirichletBC(V_d, Constant(0.0), left)
# bc_d_right = DirichletBC(V_d, Constant(0.0), right)
# bc_d_rear = DirichletBC(V_d, Constant(0.0), rear)
# bc_d_front = DirichletBC(V_d, Constant(0.0), front)

bc_d = [DirichletBC(V_d, Constant(0.0), rear)]
# bc_d = [bc_u_rear, bc_d_front, bc_u_left, bc_u_right]

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

# u0 = interpolate(zero_v, V_u)
# v0 = interpolate(zero_v, V_u)
# a0 = interpolate(zero_v, V_u)

u0 = Function(V_u)
v0 = Function(V_u)
a0 = Function(V_u)

u0.assign(zero_v)
v0.assign(zero_v)
a0.assign(zero_v)

# ipdb.set_trace()


# To define initial condition for v0
# class InitialCondition(Expression):
#     def eval_cell(self, value, x, ufc_cell):
#         if x[1] <= a and abs(x[0] - 0.) < DOLFIN_EPS:
#             value[0] = 1.0
#         else:
#             value[0] = 0.0


# v_.interpolate(InitialCondition())
# ipdb.set_trace()


# update v
def update_a(u_, u0, v0, a0, ufl=True):
    return 4.0/deltaT * (u_ - u0 - v0 * deltaT - deltaT**2.0/4.0 * a0)


def update_v(u_, u0, v0, a0, ufl=True):
    return v0 + deltaT/2.0 * a0 + deltaT/2.0 * update_a(u_, u0, v0, a0)


d0 = interpolate(Constant(0.0), V_d)


def update_fields(u_, u0, v0, a0, ufl=True):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec, u0_vec = u_.vector(), u0.vector()
    v0_vec, a0_vec = v0.vector(), a0.vector()

    # use update functions using vector arguments
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=True)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=True)

    # Update (u_old <- u)
    v0.vector()[:], a0.vector()[:] = v_vec, a_vec
    # u0.vector()[:] = u_.vector()
    
# ipdb.set_trace()


# weak forms
E_u = rho * inner(a0, u_) * dx + inner((1-d_)**2 * sigma(u_), epsilon_e(u_)) * dx - inner(traction, u_) * ds(6)
E_d = 1.0/(4.0 * cw) * Gc * (d_**2.0/ell * dx + ell * inner(grad(d_), grad(d_)) * dx)

Pi = E_u + E_d
# ipdb.set_trace()

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
# ipdb.set_trace()


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

# ipdb.set_trace()

# Initialization of the iterative procedure and output requests
min_step = 0
max_step = 0.2
n_step = 201
load_multipliers = np.linspace(min_step, max_step, n_step)
max_iterations = 100

tol = 1e-3
conc_u = File("./ResultsDir_Shen/u.pvd")
conc_d = File("./ResultsDir_Shen/d.pvd")
# conc_T = File ("./ResultsDir/T.pvd")

# fname = open('ForcevsDisp.txt', 'w')

# u_.vector()[:] = u0.vector()
# solver_u.solve()
# u0.vector()[:] = u_.vector()
#
# # d_.vector()[:] = d0.vector()
# solver_d.solve(problem_d, d_.vector(), d_lb.vector(), d_ub.vector())
# d0.vector()[:] = d_.vector()

# solver_T.solve()
# T0.vector()[:] = T_.vector()

# conc_u << a0
# conc_d << d_
# S0 = project(sym(grad(u_)), V_s)
# conc_d << S0

# u0.vector()[:] = u_.vector()
# def tr_p(trS):
#     return trS > 0.

# ipdb.set_trace()

# Staggered scheme
for (i_p, p) in enumerate(load_multipliers):

    itr = 0
    err = 1.0
    # load_top.t = 1.0e-6 * p
    # load_bot.t = 1.0e-6 * p
    # print('Load: ', load_top.t)
    v_p.t = p
    print(p)
    # ipdb.set_trace()

    while err > tol and itr < max_iterations:
        itr += 1
    #
        solver_u.solve()
        solver_d.solve(problem_d, d_.vector(), d_lb.vector(), d_ub.vector())
    # #
        err_u = errornorm(u_, u0, norm_type='l2', mesh=None)
        err_d = errornorm(d_, d0, norm_type='l2', mesh=None)
    # #
        err = max(err_u, err_d)
    #
        u0.vector()[:] = u_.vector()
        d0.vector()[:] = d_.vector()
    #
        # a_new = update_a(u_, u0, v0, a0)
        # v_new = update_v(u_, u0, v0, a0)
        # a0.vector()[:] = a_new
        # v0.vector()[:] = v_new

        update_fields(u_, u0, v0, a0, ufl=True)
        
        # ipdb.set_trace()

    #
        if err < tol:
            print ('Iterations:', itr, ', Total time', p)
            conc_d << d_
            conc_u << u_
    #         # S0 = project(sigma_n(u_, T_), V_s)
    #         # trS = project(abs(tr(sigma_p(u_, T_))) - tr(sigma_p(u_, T_)), V_d)
    #         # # trP = tr_p(trS)
    #         # conc_T << trS
    #
    # #
    # #     # Traction = dot(sigma(u_, T_), n)
    # #     # fy = Traction[1]*ds(1)
    # #
    # #     # fname.write(str(p*u_r) + "\t")
    # #     # fname.write(str(assemble(fy)) + "\n")
    #
    # # itr = 0
    # # err = 1.0
    # # while err > tol and itr < max_iterations:
    # #     itr += 1
    # #     solver_T.solve()
    # #     err_T = errornorm(T_, T0, norm_type = 'l2', mesh = None)
    # # # #
    # # # #     # # # T0.assign(T_)
    # #     T0.vector()[:] = T_.vector()
    # # # #     if err < tol:
    # # # #         print ('Iterations:', itr, ', Total time', p)
    # # conc_T << T_

# fname.close()
print ('Simulation completed')