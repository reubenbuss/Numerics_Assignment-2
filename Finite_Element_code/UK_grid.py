import matplotlib.pyplot as plt
import numpy as np
import Finite_Element_code.solver_2d as s2d

nodes = np.loadtxt(r'Grid\esw_nodes_100k.txt')
IEN = np.loadtxt(r'Grid\esw_IEN_100k.txt',
                 dtype=np.int64)
boundary_nodes = np.loadtxt(r'Grid\esw_bdry_100k.txt', dtype=np.int64)
ID = np.zeros(len(nodes), dtype=np.int64)


def gauss_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude=1):
    return amplitude * np.exp(-(((x - x0)**2) / (2 * sigma_x**2) + ((y - y0)**2) / (2 * sigma_y**2)))


def S(X):
    X = np.atleast_2d(X)
    x = X[:, 0]
    y = X[:, 1]
    x0, y0 = 442365, 115483
    sigma_x, sigma_y = 1000.0, 1000.0
    z = gauss_2d(x, y, x0, y0, sigma_x, sigma_y)
    return z


def L2NORM(analytic, numerical):
    return abs(analytic-numerical)

# boundary_nodes = range(19,35) # right coast
# boundary_nodes = [0,1,2,3,4,5,6,7,8,9,10,11,41,42,43,44,45,46] # left coast
# boundary_nodes=[41] # left most point
# boundary_nodes=[31] # right most point


boundary_conditions = {
    "Dirichlet": boundary_nodes,  # Dirichlet boundary nodes
    "Neumann": []   # Neumann boundary nodes
}

n_eq = 0
for i in range(len(nodes[:, 1])):
    if i in boundary_conditions["Dirichlet"]:  # Dirichlet boundary
        ID[i] = -1  # Exclude from equations, as values are set explicitly
    elif i in boundary_conditions["Neumann"]:  # Neumann boundary
        ID[i] = n_eq  # Include in equations, with special treatment for F
        n_eq += 1
    else:  # Interior nodes
        ID[i] = n_eq
        n_eq += 1


N_equations = np.max(ID)+1
N_elements = IEN.shape[0]
N_nodes = nodes.shape[0]
N_dim = nodes.shape[1]
# Location matrix
LM = np.zeros_like(IEN.T)
for e in range(N_elements):
    for a in range(3):
        LM[a, e] = ID[IEN[e, a]]
# Global stiffness matrix and force vector
K = np.zeros((N_equations, N_equations))
F = np.zeros((N_equations,))
# Loop over elements
for e in range(N_elements):
    k_e = s2d.stiffness_matrix_for_element(nodes[IEN[e, :], :])
    f_e = s2d.force_vector_for_element(nodes[IEN[e, :], :], S)
    for a in range(3):
        A = LM[a, e]
        for b in range(3):
            B = LM[b, e]
            if (A >= 0) and (B >= 0):  # Interior node equations
                K[A, B] += k_e[a, b]
        if (A >= 0):  # Include Neumann BC contributions
            F[A] += f_e[a]
            if IEN[e, a] in boundary_conditions["Neumann"]:
                F[A] += 0       # neumann_flux_value(nodes[IEN[e, a]]


# Solve
Psi_interior = np.linalg.solve(K, F)
Psi_A = np.zeros(N_nodes)
for n in range(N_nodes):
    # Otherwise Psi should be zero, and we've initialized that already.
    if ID[n] >= 0:
        Psi_A[n] = Psi_interior[ID[n]]

for n in boundary_conditions["Dirichlet"]:
    # Psi_A[n] = exact_point(nodes[n,:])  # Explicitly set the value
    Psi_A[n] = 0

plt.tripcolor(nodes[:, 0], nodes[:, 1], Psi_A, triangles=IEN)
# identify the boundary nodes are placed correctly
# plt.scatter(nodes[boundary_nodes, 0], nodes[boundary_nodes, 1], color='red', label='Boundary Nodes',marker='.')
plt.scatter(nodes[boundary_conditions["Dirichlet"], 0],
            nodes[boundary_conditions["Dirichlet"], 1],
            color='orange', label='Dirichlet Boundary Nodes')
plt.scatter(nodes[boundary_conditions["Neumann"], 0],
            nodes[boundary_conditions["Neumann"], 1],
            color='lightblue', label='Neumann Boundary Nodes')
plt.scatter([442365, 473993], [115483, 171625], c='k')
n = 0
# plt.scatter(nodes[n,0],nodes[n,1],c='b',label='Selector',marker='*')
plt.colorbar()
plt.legend()
plt.show()
