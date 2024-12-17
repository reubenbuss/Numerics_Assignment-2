import matplotlib.pyplot as plt
import numpy as np
import Finite_Element_code.solver_2d as s2d


def south_grid(res):
    nodes = np.loadtxt(r'Grid\las_nodes_40k.txt')
    IEN = np.loadtxt(r'Grid\las_IEN_40k.txt',
                     dtype=np.int64)
    boundary_nodes = np.loadtxt(r'Grid\las_bdry_40k.txt', dtype=np.int64)
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
    x2, y2 = tri[2, :, 0], tri[1, :, 1]
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


def create_boundary_conditions(nodes, ID):
    boundary_conditions = {
        # Dirichlet boundary nodes
        "Dirichlet": [0, 1, 2, 3, 4, 26, 27, 28, 29, 30, 31],
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
    N_equations = np.max(ID)+1
    N_elements = IEN.shape[0]

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
        windspeed = 0.00005
        k_e = s2d.stiffness_matrix_for_element(nodes[IEN[e, :], :])
        f_e = s2d.force_vector_for_element(nodes[IEN[e, :], :], S)
        a_e = s2d.advection_matrix_for_element(nodes[IEN[e, :], :], windspeed)
        for a in range(3):
            A = LM[a, e]
            for b in range(3):
                B = LM[b, e]
                if (A >= 0) and (B >= 0):  # Interior node equations
                    K[A, B] += k_e[a, b] - a_e[a, b]
                    # print(f'K:{K[A,B]}')
                    # print(f'k_e:{k_e[a,b]}')
                    # print(f'a_e:{a_e[a,b]}')
            if (A >= 0):  # Include Neumann BC contributions
                F[A] += f_e[a]
                if IEN[e, a] in boundary_conditions["Neumann"]:
                    F[A] += 0       # neumann_flux_value(nodes[IEN[e, a]]
    print(f'Windspeed: {windspeed}')
    return K, F


def solve_for_psi(K, F, ID, nodes, boundary_conditions):
    N_nodes = nodes.shape[0]
    Psi_interior = np.linalg.solve(K, F)
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
    plt.scatter(tri[:, 0], tri[:, 1], c='b')
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
    plt.scatter([473993], [171625], marker='o',
                facecolor='None', edgecolor='black', label='Reading')

    n = 52
    plt.scatter(nodes[n, 0], nodes[n, 1], c='b', label='Selector', marker='*')
    plt.colorbar()
    # plt.legend()
    plt.show()


def time_dependent_solver(nodes, IEN, initial_condition, ID, source_func, dt, num_steps, boundary_conditions):
    N_equations = np.max(ID)+1
    N_elements = IEN.shape[0]

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
        windspeed = 0.00005
        m_e = s2d.mass_matrix_for_element(nodes[IEN[e, :], :])
        k_e = s2d.stiffness_matrix_for_element(nodes[IEN[e, :], :])
        f_e = s2d.force_vector_for_element(nodes[IEN[e, :], :], S)
        a_e = s2d.advection_matrix_for_element(nodes[IEN[e, :], :], windspeed)
        for a in range(3):
            A = LM[a, e]
            for b in range(3):
                B = LM[b, e]
                if (A >= 0) and (B >= 0):  # Interior node equations
                    K[A, B] += k_e[a, b] - a_e[a, b]
                    # print(f'K:{K[A,B]}')
                    # print(f'k_e:{k_e[a,b]}')
                    # print(f'a_e:{a_e[a,b]}')
            if (A >= 0):  # Include Neumann BC contributions
                F[A] += f_e[a]
                if IEN[e, a] in boundary_conditions["Neumann"]:
                    F[A] += 0       # neumann_flux_value(nodes[IEN[e, a]]
    print(f'Windspeed: {windspeed}')
    return K, F


def run():
    res = 52
    nodes, IEN, boundary_nodes, ID = south_grid(res)
    boundary_conditions, ID2 = create_boundary_conditions(nodes, ID)
    K, F = compute_matrices(nodes, IEN, ID2, boundary_conditions)
    Psi_A = solve_for_psi(K, F, ID2, nodes, boundary_conditions)
    plot_results(nodes, IEN, boundary_conditions, boundary_nodes, Psi_A)


# run()

def time_run():
    res = 40
    nodes, IEN, boundary_nodes, ID = south_grid(res)
    boundary_conditions, ID = create_boundary_conditions(nodes, ID)
    Psi_A = time_dependent_solver(
        nodes, IEN, S, ID, S, 1/100, 10, boundary_conditions)
    plot_results(nodes, IEN, boundary_conditions, boundary_nodes, Psi_A)


time_run()
