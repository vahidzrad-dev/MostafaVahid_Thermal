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

L = 100.0e-3	# Width: mm (Chu 2017-3.3)
H = 40.0e-3	    # Height: mm (Chu 2017-3.3)
a = 50.0e-3     # Crack length

subdir = "meshes/"
meshname = "mesh" # "fracking_hsize%g" % (hsize)

mesh = Mesh(subdir + meshname + ".xml")

# Define Space
V_u = VectorFunctionSpace(mesh, 'CG', 1)
V_d = FunctionSpace(mesh, 'CG', 1)
V_T = FunctionSpace(mesh, 'CG', 1)

u_, u, u_t = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
d_, d, d_t = Function(V_d), TrialFunction(V_d), TestFunction(V_d)
T_, T, T_t = Function(V_T), TrialFunction(V_T), TestFunction(V_T)

# Introduce manually the material parameters
E = 32.0e9		# Young's modulus: MPa (Chu 2017-4.1)
nu = 0.2		# Poisson's ratio: - (Chu 2017-4.1)
Gc = 2.7e3		# critical energy release rate: MPa-mm (Chu 2017-4.1)

hsize = 1.25e-4       # mesh size: mm (Chu 2017-4.1)
l = 2. * hsize      # length scale: mm (Chu 2017-4.1)

Ts = Constant(680.)  	# initial temperature of slab: K (Chu 2017-3.3)
Tw = Constant(300.)	# temperature of surface contacted with water: K (Chu 2017-3.3)

lmbda  = Constant(E*nu/((1+nu)*(1-2*nu)))		# Lamé constant: MPa (conversion formulae)
mu = Constant(E/(2*(1+nu))) 				# shear modulus: MPa (conversion formulae)

rho = 2.45e3						# density: kg/m^3 (Chu 2017-4.1)

beta = Constant(6.6e-6)				# linear expansion coefficient: 1/K (Chu 2017-4.1)

c = Constant(961.5e6)					# specific heat of material: #J/(kgK) (Chu 2017-4.1)
k = Constant(21.0e3)					# thermal conductivity: #W/(mK)=J/(mKs) (Chu 2017-4.1)
deltaT = rho * c * hsize**2 / k				# source: (Chu2017-3.3: hsize vs H?)
print('DeltaT', deltaT)

# Constituive functions
# strain
def epsilon(u_):
    return sym(grad(u_))

def epsilonT(u_, T_):
    return beta * (T_ - Ts) * Identity(len(u_))

# def epsilone(u_, T_):
#     return epsilon(u_) - epsilonT(u_, T_)

def epsilone(u_, T_):
    return sym(grad(u_))

# stress
# def sigma(u_): # not applicable
#     return lmbda * tr(epsilon(u_)) * Identity(len(u_)) + 2.0 * mu * epsilon(u_)

def sigma(u_, T_): # no decomposition
    return lmbda * tr(epsilone(u_, T_)) * Identity(len(u_)) + 2.0 * mu * (epsilone(u_, T_))

def sigmap(u_, T_): # sigma_+ for Amor's model
    return (lmbda + 2.0 * mu / 3.0) * (tr(epsilone(u_, T_)) + abs(tr(epsilone(u_, T_))))/2.0 * Identity(len(u_)) \
    + 2.0 * mu * dev(epsilone(u_, T_))
def sigman(u_, T_): # sigma_- for Amor's model
    return (lmbda + 2.0 * mu / 3.0) * (tr(epsilone(u_, T_)) - abs(tr(epsilone(u_, T_))))/2.0 * Identity(len(u_))

# strain energy
def psi(u_, T_):
    return lmbda/2.0 * tr(epsilone(u_, T_))**2 + mu * inner(epsilone(u_, T_), epsilone(u_, T_))

def psip(u_, T_):
    return (lmbda/2.0 + mu/3.0) * ((tr(epsilone(u_, T_)) + abs(tr(epsilone(u_, T_))))/2.0)**2 \
    + mu * inner(dev(epsilone(u_, T_)), dev(epsilone(u_, T_)))

def psin(u_, T_):
    return (lmbda/2.0 + mu/3.0) * ((tr(epsilone(u_, T_)) - abs(tr(epsilone(u_, T_))))/2.0)**2

# Boundary conditions
top = CompiledSubDomain("near(x[1], 5.0) && on_boundary")
bot = CompiledSubDomain("near(x[1], -5.0) && on_boundary")
left = CompiledSubDomain("near(x[0], -5.0) && on_boundary")
right = CompiledSubDomain("near(x[0], 5.0) && on_boundary")

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

load_top = Expression("t", t = 0.0, degree=1)
load_bot = Expression("-t", t = 0.0, degree=1)

# Boundary conditions for u
bc_u_top = DirichletBC(V_u.sub(1), load_top, top)
bc_u_bot_i = DirichletBC(V_u.sub(1), load_bot, bot)
bc_u_bot_ii = DirichletBC(V_u.sub(0), Constant(0.0), bot)
# bc_u_right = DirichletBC(V_u.sub(0), Constant(0.), right)
# bc_u_left = DirichletBC(V_u.sub(0), Constant(0.), left)
# bc_u_pt_left = DirichletBC(V_u, Constant([0.,0.]), pinpoint_l, method='pointwise')
# bc_u_pt_right = DirichletBC(V_u, Constant([0.,0.]), pinpoint_r, method='pointwise')
bc_u = [bc_u_top, bc_u_bot_i, bc_u_bot_ii]

def Crack(x):
    return abs(x[1]) < 1e-03 and x[0] <= a/2.0 and x[0] >= -a/2.0

bc_d = [DirichletBC(V_d, Constant(1.0), Crack)]

# Boundary conditions for T
bc_T = [DirichletBC(V_T, Tw, bot)]
# bc_T_top = DirichletBC(V_T, Ts, top)
# bc_T_right = DirichletBC(V_T, Ts, right)
# bc_T_left = DirichletBC(V_T, Ts, left)
# bc_T = [bc_T_top, bc_T_right, bc_T_left]

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top.mark(boundaries,1)
bot.mark(boundaries,2)
ds = Measure("ds")(subdomain_data = boundaries)
n = FacetNormal(mesh)

# weak form
# E_u = (1.0-d_)**2.0 * inner(sigmap(u, T_), epsilone(u_t, T_)) * dx + inner(sigman(u, T_), epsilone(u_t, T_)) * dx
# E_d = (3.0/8.0) * Gc * (d_t/l * dx + 2.0 * l * inner(grad(d), grad(d_t)) * dx) - 2.0 * (1.0 - d) * psip(u_, T_) * d_t * dx

zero_v = Constant((0.,)*2)
u0 = interpolate(zero_v, V_u)

d0 = interpolate(Constant(0.0), V_d)
T0 = interpolate(Expression('T_init', T_init = Ts, degree=1), V_T)

E_u = (1.0 - d_)**2.0 * psip(u_, T_) * dx + psin(u_, T_) * dx
E_d = (1.0/2.0) * Gc * (d_**2/l * dx + l * inner(grad(d_), grad(d_)) * dx)
# E_d = (3.0/8.0) * Gc * (d_/l * dx + l * inner(grad(d_), grad(d_)) * dx)
E_T = (1.0 - d_)**2 * rho * c * (T_ - T0) * T * dx - deltaT * (1.0 - d_)**2 * k * inner(grad(T_), grad(T)) * dx

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

J_T  = derivative(E_T, T_, T_t)
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
max_step = 95.0e-7
n_step = 760
load_multipliers = np.linspace(min_step, max_step, n_step)
max_iterations = 100

tol = 1e-3
conc_u = File ("./ResultsDir/u.pvd")
conc_d = File ("./ResultsDir/d.pvd")
# conc_T = File ("./ResultsDir/T.pvd")

# fname = open('ForcevsDisp.txt', 'w')

# ur = 0.05
solver_u.solve()
u0.vector()[:] = u_.vector()

# d_.vector()[:] = d0.vector()
solver_d.solve(problem_d, d_.vector(), d_lb.vector(), d_ub.vector())
d0.vector()[:] = d_.vector()

# T_.vector()[:] = T0.vector()
# solver_T.solve()
# T_.vector()[:] = T0.vector()
# conc_T << T0

# Staggered scheme
for (i_p, p) in enumerate(load_multipliers):

    iter = 0
    err = 1.0
    load_top.t = 1.0e-3 * p
    load_bot.t = 1.0e-3 * p
    print('Load: ', load_top.t)
    
    while err > tol and iter < max_iterations:
        iter += 1
        solver_u.solve()
        solver_d.solve(problem_d, d_.vector(), d_lb.vector(), d_ub.vector())
        # solver_T.solve()
    #
        err_u = errornorm(u_, u0, norm_type = 'l2', mesh = None)
        err_d = errornorm(d_, d0, norm_type = 'l2', mesh = None)
    #     err_T = errornorm(T_, T0, norm_type = 'l2', mesh = None)
    #
        err = max(err_d, err_u)
        # err = err_d
        # print('err_u: ', err_u)
        
        u0.vector()[:] = u_.vector()
        d0.vector()[:] = d_.vector()
    
        if err < tol:
            print ('Iterations:', iter, ', Total time', p)
            conc_d << d_
            conc_u << u_
    #
    #     # Traction = dot(sigma(u_, T_), n)
    #     # fy = Traction[1]*ds(1)
    #
    #     # fname.write(str(p*u_r) + "\t")
    #     # fname.write(str(assemble(fy)) + "\n")
    
    # iter = 0
    # err = 1.0
    # while err > tol and iter < max_iterations:
    #     iter += 1
        solver_T.solve()
    #     err_T = errornorm(T_, T0, norm_type = 'l2', mesh = None)
    #
    #     # # # T0.assign(T_)
        T0.vector()[:] = T_.vector()
    #     if err < tol:
    #         print ('Iterations:', iter, ', Total time', p)
        conc_T << T_

    
# fname.close()
print ('Simulation completed')