import matplotlib.pyplot as plt
import numpy as np
import Finite_Element_code.solver_2d as s2d
from scipy import sparse as sp


def south_grid(res):
    nodes = np.loadtxt(r'Grid\las_nodes_5k.txt')
    IEN = np.loadtxt(r'Grid\las_IEN_5k.txt',
                     dtype=np.int64)
    boundary_nodes = np.loadtxt(r'Grid\las_bdry_5k.txt', dtype=np.int64)
    ID = np.zeros(len(nodes), dtype=np.int64)
    return nodes, IEN, boundary_nodes, ID


def uk_grid(res):
    nodes = np.loadtxt(r'Grid\esw_nodes_100k.txt')
    IEN = np.loadtxt(r'Grid\esw_IEN_100k.txt',
                     dtype=np.int64)
    boundary_nodes = np.loadtxt(r'Grid\esw_bdry_100k.txt', dtype=np.int64)
    ID = np.zeros(len(nodes), dtype=np.int64)
    return nodes, IEN, boundary_nodes, ID


def gauss_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude=1):
    return amplitude * np.exp(-(((x - x0)**2) / (2 * sigma_x**2) + ((y - y0)**2) / (2 * sigma_y**2)))


def area_of_triangle(tri):
    x0, y0 = tri[0, :, 0], tri[0, :, 1]
    x1, y1 = tri[1, :, 0], tri[1, :, 1]
    x2, y2 = tri[2, :, 0], tri[2, :, 1]
    return 1/2*abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))


def point_in_triangle(coord, vertices):
    array_of_coords = np.reshape(np.tile(coord, len(vertices)), (-1, 2))
    A = area_of_triangle(
        np.array([vertices[:, 0], vertices[:, 1], vertices[:, 2]]))
    A1 = area_of_triangle(
        np.array([vertices[:, 0], vertices[:, 1], array_of_coords]))
    A2 = area_of_triangle(
        np.array([vertices[:, 0], vertices[:, 2], array_of_coords]))
    A3 = area_of_triangle(
        np.array([vertices[:, 1], vertices[:, 2], array_of_coords]))
    return np.argmin(abs(A-(A1+A2+A3)))


def find_Reading(nodes, IEN):
    reading = np.array([473993, 171625])
    nearest_point = np.argmin(
        (nodes[:, 0] - 473993)**2+(nodes[:, 1] - 171625)**2, axis=0)
    triangles = np.where(nearest_point == IEN)[0]
    vertices = nodes[IEN[triangles]]
    return nodes[IEN[triangles[point_in_triangle(reading, vertices)]]]


def S(X):
    X = np.atleast_2d(X)
    x = X[:, 0]
    y = X[:, 1]
    x0, y0 = 442365, 115483
    sigma_x, sigma_y = 1000.0, 1000.0
    z = gauss_2d(x, y, x0, y0, sigma_x, sigma_y)
    return z


def NORM(analytic, numerical):
    return abs(analytic-numerical)


def create_boundary_conditions(nodes, ID, boundary_nodes):
    southern_border = np.where(nodes[boundary_nodes, 1] <= 110000)[0]
    boundary_conditions = {
        # Dirichlet boundary nodes
        "Dirichlet": [0, 1, 2, 3, 4, 26, 27, 28, 29, 30, 31],
        "Neumann": []   # Neumann boundary nodes
    }
    boundary_conditions = {
        # Dirichlet boundary nodes
        "Dirichlet": southern_border,
        "Neumann": []   # Neumann boundary nodes
    }
    # boundary_conditions = {
    #     # Dirichlet boundary nodes
    #     "Dirichlet": [34,35,36,37,38,39,40,41],
    #     "Neumann": []   # Neumann boundary nodes
    # }

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
    return boundary_conditions, ID


def compute_matrices(nodes, IEN, ID, boundary_conditions):
    N_equations = np.max(ID) + 1
    N_elements = IEN.shape[0]

    LM = np.zeros_like(IEN.T)
    for e in range(N_elements):
        for a in range(3):
            LM[a, e] = ID[IEN[e, a]]

    K = sp.lil_matrix((N_equations, N_equations))  # Stiffness matrix
    M = sp.lil_matrix((N_equations, N_equations))  # Mass matrix
    A = sp.lil_matrix((N_equations, N_equations))  # Advection matrix
    F = np.zeros((N_equations,))                   # Force vector

    for e in range(N_elements):
        wind_velocity = [-5, -20]
        k_e = s2d.stiffness_matrix_for_element(nodes[IEN[e, :], :])
        m_e = s2d.mass_matrix_for_element(nodes[IEN[e, :], :])
        a_e = s2d.advection_matrix_for_element(
            nodes[IEN[e, :], :], wind_velocity)
        f_e = s2d.force_vector_for_element(nodes[IEN[e, :], :], S)

        for a in range(3):
            A_index = LM[a, e]
            for b in range(3):
                B_index = LM[b, e]
                if A_index >= 0 and B_index >= 0:
                    K[A_index, B_index] += k_e[a, b]
                    M[A_index, B_index] += m_e[a, b]
                    A[A_index, B_index] += a_e[a, b]
            if A_index >= 0:
                F[A_index] += f_e[a][0]

    return M, K, A, F


def solve_for_psi_time(M, K, A, F, ID, nodes, boundary_conditions, dt, T):
    N_nodes = nodes.shape[0]
    N_equations = np.max(ID) + 1

    # Initial condition (set psi_0 to zero or some initial condition)
    Psi_n = np.zeros(N_nodes)

    # Implicit Euler System Matrix
    system_matrix = M + dt * (K - A)

    # Time-stepping loop
    for t in np.arange(0, T, dt):
        # Right-hand side: M * Psi^n + dt * F
        rhs = M @ Psi_n[ID >= 0] + dt * F

        # Solve the linear system
        Psi_interior = sp.linalg.spsolve(sp.csr_matrix(system_matrix), rhs)

        # Update the solution
        Psi_n[ID >= 0] = Psi_interior

        # Apply Dirichlet boundary conditions (if needed)
        for n in boundary_conditions["Dirichlet"]:
            Psi_n[n] = 0

    return Psi_n


def solve_for_psi(K, F, ID, nodes, boundary_conditions):
    N_nodes = nodes.shape[0]
    K = sp.csr_matrix(K)
    Psi_interior = sp.linalg.spsolve(K, F)
    Psi_A = np.zeros(N_nodes)
    for n in range(N_nodes):
        # Otherwise Psi should be zero, and we've initialized that already.
        if ID[n] >= 0:
            Psi_A[n] = Psi_interior[ID[n]]

    for n in boundary_conditions["Dirichlet"]:
        # Psi_A[n] = exact_point(nodes[n,:])  # Explicitly set the value
        Psi_A[n] = 0
    return Psi_A


def plot_results(nodes, IEN, boundary_conditions, boundary_nodes, Psi_A):
    tri = find_Reading(nodes, IEN)
    plt.tripcolor(nodes[:, 0], nodes[:, 1], Psi_A, triangles=IEN)
    # plt.scatter(tri[:, 0], tri[:, 1], c='b') # around reading
    # identify the boundary nodes are placed correctly
    plt.scatter(nodes[boundary_nodes, 0], nodes[boundary_nodes, 1],
                color='red', label='Boundary Nodes', marker='.')
    plt.scatter(nodes[boundary_conditions["Dirichlet"], 0],
                nodes[boundary_conditions["Dirichlet"], 1],
                color='orange', label='Dirichlet Boundary Nodes')
    plt.scatter(nodes[boundary_conditions["Neumann"], 0],
                nodes[boundary_conditions["Neumann"], 1],
                color='lightblue', label='Neumann Boundary Nodes')
    plt.scatter([442365], [115483], marker='o',
                facecolor='None', edgecolor='black', label='Southampton')
    plt.scatter([473993], [171625], marker='.',
                facecolor='None', edgecolor='black', label='Reading')

    n = 52
    # plt.scatter(nodes[n, 0], nodes[n, 1], c='b', label='Selector', marker='*')
    plt.colorbar()
    # plt.legend()
    plt.show()


def run():
    res = 52
    nodes, IEN, boundary_nodes, ID = south_grid(res)
    boundary_conditions, ID2 = create_boundary_conditions(
        nodes, ID, boundary_nodes)
    K, F = compute_matrices(nodes, IEN, ID2, boundary_conditions)
    Psi_A = solve_for_psi(K, F, ID2, nodes, boundary_conditions)
    plot_results(nodes, IEN, boundary_conditions, boundary_nodes, Psi_A)


def run_time():
    res = 52
    nodes, IEN, boundary_nodes, ID = south_grid(res)
    boundary_conditions, ID2 = create_boundary_conditions(
        nodes, ID, boundary_nodes)

    M, K, A, F = compute_matrices(nodes, IEN, ID2, boundary_conditions)

    dt = 10      # Time step
    T = 1000     # Total simulation time
    Psi_A = solve_for_psi_time(
        M, K, A, F, ID2, nodes, boundary_conditions, dt, T)

    plot_results(nodes, IEN, boundary_conditions, boundary_nodes, Psi_A)


run_time()
