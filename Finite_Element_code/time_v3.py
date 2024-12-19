import numpy as np
from scipy import sparse as sp
from matplotlib.tri import Triangulation
from PIL import Image
import matplotlib.pyplot as plt
import solver_2d as s2d
import time


def south_grid(res):
    """
    Load the southern grid's Nodes: location of the nodes of the grid,
    IEN: the mapping from global element number to global node numbers,
    boundary_nodes: which global node numbers lie on the boundary.

    Args:
        res (str): Resolution of the grid, typically represented as a string (e.g., '5').

    Returns:
        tuple: A tuple containing:
            - nodes (numpy.ndarray): Array of node coordinates.
            - IEN (numpy.ndarray): Element connectivity array.
            - boundary_nodes (numpy.ndarray): Array of boundary node indices.
            - ID (numpy.ndarray): Array for node indexing.
    """

    nodes = np.loadtxt('Grid/las_nodes_'+res+'k.txt')
    IEN = np.loadtxt('Grid/las_IEN_'+res+'k.txt',
                     dtype=np.int64)
    boundary_nodes = np.loadtxt('Grid/las_bdry_'+res+'k.txt', dtype=np.int64)
    ID = np.zeros(len(nodes), dtype=np.int64)
    return nodes, IEN, boundary_nodes, ID


def uk_grid(res):
    """
    Load the UK grid's Nodes: location of the nodes of the grid,
    IEN: the mapping from global element number to global node numbers,
    boundary_nodes: which global node numbers lie on the boundary.

    Args:
        res (str): Resolution of the grid, typically represented as a string (e.g., '100').

    Returns:
        tuple: A tuple containing:
            - nodes (numpy.ndarray): Array of node coordinates.
            - IEN (numpy.ndarray): Element connectivity array.
            - boundary_nodes (numpy.ndarray): Array of boundary node indices.
            - ID (numpy.ndarray): Array for node indexing.
    """
    nodes = np.loadtxt(r'Grid\esw_nodes_100k.txt')
    IEN = np.loadtxt(r'Grid\esw_IEN_100k.txt',
                     dtype=np.int64)
    boundary_nodes = np.loadtxt(r'Grid\esw_bdry_100k.txt', dtype=np.int64)
    ID = np.zeros(len(nodes), dtype=np.int64)
    return nodes, IEN, boundary_nodes, ID


def gauss_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude=1):
    """
    Compute a 2D Gaussian function.

    Args:
        x (float or numpy.ndarray): X-coordinate(s).
        y (float or numpy.ndarray): Y-coordinate(s).
        x0 (float): X-coordinate of the Gaussian center.
        y0 (float): Y-coordinate of the Gaussian center.
        sigma_x (float): Standard deviation in the X direction.
        sigma_y (float): Standard deviation in the Y direction.
        amplitude (float, optional): Amplitude of the Gaussian. Defaults to 1.

    Returns:
        numpy.ndarray: Gaussian values at the given coordinates.
    """
    return amplitude * np.exp(-(((x - x0)**2) / (2 * sigma_x**2) + ((y - y0)**2) / (2 * sigma_y**2)))


def area_of_triangle(tri):
    """
    Calculate the area of a triangle given its vertices.

    Args:
        tri (numpy.ndarray): Array of shape (3, 2) containing triangle vertex coordinates.

    Returns:
        float: Area of the triangle.
    """
    x0, y0 = tri[0, :, 0], tri[0, :, 1]
    x1, y1 = tri[1, :, 0], tri[1, :, 1]
    x2, y2 = tri[2, :, 0], tri[2, :, 1]
    return 1/2*abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))


def point_in_triangle(coord, vertices):
    """
    Determine the triangle containing a given point based using area test.

    Args:
        coord (numpy.ndarray): Coordinates of the point (x, y).
        vertices (numpy.ndarray): Vertices of the triangles (shape: n_triangles x 3 x 2).

    Returns:
        int: Index of the triangle that contains the given point.
    """
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
    """
    Locate the triangle containing Reading in the grid.

    Args:
        nodes (numpy.ndarray): Array of node coordinates.
        IEN (numpy.ndarray): Element connectivity array.

    Returns:
        numpy.ndarray: Element connectivity of the triangle containing Reading.
    """
    Reading = np.array([473993, 171625])
    nearest_point = np.argmin(
        (nodes[:, 0] - 473993)**2+(nodes[:, 1] - 171625)**2, axis=0)
    triangles = np.where(nearest_point == IEN)[0]
    vertices = nodes[IEN[triangles]]
    return IEN[triangles[point_in_triangle(Reading, vertices)]]


def weighting_at_Reading(tri):
    """
    Compute weights of Reading's position relative to a triangle using the basis functions.

    Args:
        tri (numpy.ndarray): Array of shape (3, 2) representing triangle vertex coordinates.

    Returns:
        numpy.ndarray: Basis function weights for Reading within the triangle.
    """
    x, y = 473993, 171625
    x0, x1, x2 = tri[:, 0]
    y0, y1, y2 = tri[:, 1]
    a = np.array([[x1-x0, x2-x0], [y1-y0, y2-y0]])
    b = np.array([x-x0, y-y0])
    return np.linalg.solve(a, b)


def Psi_at_reading(psi_values, tri):
    """
    Calculate pollutant concentration at Reading from the 3 nodes defining the triangle containig Reading and weighting using the basis functions 

    Args:
        psi_values (numpy.ndarray): Pollutant concentrations at triangle vertices.
        tri (numpy.ndarray): Triangle vertex coordinates.

    Returns:
        numpy.ndarray: Weighted pollutant concentration at Reading.
    """
    xi = weighting_at_Reading(tri)
    weights = [1 - xi[0] - xi[1], xi[0], xi[1]]
    concentration = psi_values[:, 0]*weights[0] + \
        psi_values[:, 1]*weights[1] + psi_values[:, 2]*weights[2]
    return np.array(concentration)


def Psi_at_reading_plot(psi_values, tri, dt, T, file_name):
    """
    Plot pollutant concentration at Reading over time.

    Args:
        psi_values (numpy.ndarray): Pollutant concentrations at triangle vertices over time.
        tri (numpy.ndarray): Triangle vertex coordinates.
        dt (int): Time step size.
        T (int): Total simulation time.
        file_name (str): Filename for saving the plot.

    Saves:
        Plot of pollutant concentration at Reading over time.
    """
    xi = weighting_at_Reading(tri)
    weights = [1 - xi[0] - xi[1], xi[0], xi[1]]
    concentration = psi_values[:, 0]*weights[0] + \
        psi_values[:, 1]*weights[1] + psi_values[:, 2]*weights[2]
    plt.plot(range(0, T, dt), concentration)
    # plt.xticks(range(0, T, 3600), range(0, T//3600))
    plt.xlabel('Time since start of fire (hours)')
    plt.ylabel(
        'Ratio of concentration of pollutant above Reading \n compared to Southampton')
    plt.savefig(f'Plots2/Psi_over_Reading{file_name}.pdf', bbox_inches="tight")


def normalize_psi(Psi_frames):
    """
    Normalize Psi values across all frames.

    Args:
        Psi_frames (numpy.ndarray): Array of Psi values for each time step.

    Returns:
        tuple: A tuple containing:
            - normalized_frames (numpy.ndarray): Normalized Psi values.
            - psi_min (float): Minimum Psi value.
            - psi_max (float): Maximum Psi value.
    """
    Psi_frames = np.where(Psi_frames < 0, 0, Psi_frames)
    all_psi_values = np.concatenate(Psi_frames)
    psi_min = np.min(all_psi_values)
    psi_max = np.max(all_psi_values)
    normalized_frames = np.array([(Psi-psi_min) / (psi_max-psi_min)
                                  for Psi in Psi_frames])
    return normalized_frames, psi_min, psi_max


def S(X):
    """
    Define the source function as a 2D Gaussian around Southampton

    Args:
        X (numpy.ndarray): Coordinates where the source function is evaluated.

    Returns:
        numpy.ndarray: Source function values at the given coordinates.
    """
    X = np.atleast_2d(X)
    x = X[:, 0]
    y = X[:, 1]
    x0, y0 = 442365, 115483
    sigma_x, sigma_y = 10000.0, 10000.0
    z = gauss_2d(x, y, x0, y0, sigma_x, sigma_y)
    return z


def wind_velocity(magnitude, model):
    """
    Compute the wind velocity vector based on the magnitude and model type.

    The vector is defined as pointing from Southampton to Reading, representing the direction of the wind flow. 
    The unit vector is scaled by the input magnitude

    For Crank Nicholson the wind direction is reversed as that seems to work... idk why.

    Args:
        magnitude (float): The magnitude of the wind velocity.
        model (str): The numerical model being used. If 'Crank-Nicholson', the wind direction is reversed.

    Returns:
        numpy.ndarray: The wind velocity vector [vx, vy].
    """
    Reading = np.array([473993, 171625])
    Southampton = np.array([442365, 115483])
    dx, dy = Reading-Southampton
    unit_vector = np.array([dx, dy])/(np.sqrt(dx**2+dy**2))
    if model == 'Crank-Nicholson':
        return -magnitude*unit_vector
    else:
        return magnitude*unit_vector


def NORM(analytic, numerical):
    """
    Compute the absolute difference between the analytic and numerical values.

    Args:
        analytic (float or numpy.ndarray): Analytic solution value(s).
        numerical (float or numpy.ndarray): Numerical solution value(s).

    Returns:
        float or numpy.ndarray: Absolute difference between the inputs.
    """
    return abs(analytic-numerical)


def upwind(boundary_nodes):
    """
    Identify boundary nodes that are upwind relative to the wind direction.

    Args:
        boundary_nodes (numpy.ndarray): Array of boundary node coordinates.

    Returns:
        numpy.ndarray: Indices of upwind boundary nodes.
    """
    wind_direction = wind_velocity(1, 'Backwards Euler')
    perp_to_wind = np.array([-wind_direction[1], wind_direction[0]])
    Southampton = np.array([442365, 115483])
    m = perp_to_wind[1]/perp_to_wind[0]
    c = Southampton[1] - m*Southampton[0]
    node_to_Southampton = np.array(
        [boundary_nodes[0] - Southampton[0], boundary_nodes[1] - Southampton[1]])
    cross_product = perp_to_wind[0] * node_to_Southampton[1] - \
        perp_to_wind[1] * node_to_Southampton[0]
    condition = boundary_nodes[1] - m*boundary_nodes[0]
    return np.where(boundary_nodes[:, 1] - m*boundary_nodes[:, 0] < c)[0]


def in_and_out_flow_boundaries(nodes, boundary_nodes):
    """
    Classify boundary nodes into inflow and outflow based on wind direction.

    Args:
        nodes (numpy.ndarray): Array of node coordinates.
        boundary_nodes (numpy.ndarray): Indices of boundary nodes.

    Returns:
        dict: A dictionary with:
            - "Inflow" (list): Indices of inflow boundary nodes.
            - "Outflow" (list): Indices of outflow boundary nodes.
    """
    wind_direction = wind_velocity(1, 'Backwards Euler')
    boundary_types = {"Inflow": [], "Outflow": []}
    x0, y0 = nodes[boundary_nodes[-1]]
    x1, y1 = nodes[boundary_nodes[0]]
    dx = x1-x0
    dy = y1-y0
    inward_normal = np.array([dy, -dx])
    outward_normal = np.array([-dy, dx])
    dot_product = np.dot(wind_direction, outward_normal)
    if dot_product < 0:
        boundary_types["Inflow"].append(boundary_nodes[0])
    else:
        boundary_types["Outflow"].append(boundary_nodes[0])
    for i in range(1, len(boundary_nodes)):
        x0, y0 = nodes[boundary_nodes[i-1]]
        x1, y1 = nodes[boundary_nodes[i]]
        dx = x1-x0
        dy = y1-y0
        right_normal = np.array([dy, -dx])
        left_normal = np.array([-dy, dx])
        # since the boundary conditions have been labelled in a clockwise direction the right normal should be inside pointing and left normal outwards pointing
        dot_product = np.dot(wind_direction, left_normal)
        if dot_product < 0:
            boundary_types["Inflow"].append(boundary_nodes[i])
        else:
            boundary_types["Outflow"].append(boundary_nodes[i])
    return boundary_types


def removing_irregularities_in_boundary(boundary_nodes, upwind_nodes, boundary_types):
    """
    Adjust boundary conditions to eliminate irregularities and assign Dirichlet or Neumann conditions. 
    This was a silly idea and did not help at all 

    Args:
        boundary_nodes (numpy.ndarray): Indices of boundary nodes.
        upwind_nodes (numpy.ndarray): Indices of upwind boundary nodes.
        boundary_types (dict): Dictionary classifying boundaries into inflow and outflow.

    Returns:
        dict: Updated boundary conditions with "Dirichlet" and "Neumann" classifications.
    """
    Dirichlet = np.unique(np.concatenate(
        (boundary_types["Inflow"], upwind_nodes)))
    Neumann = [x for x in boundary_types["Outflow"] if x not in upwind_nodes]
    boundary_classifier = []
    boundary_conditions = {"Dirichlet": [
        boundary_nodes[-1], boundary_nodes[-2]], "Neumann": []}
    for node in boundary_nodes:
        if node in Dirichlet:
            boundary_classifier.append(0)
        else:
            boundary_classifier.append(1)
    for i in range(0, len(boundary_nodes)-2):
        # m = 1/5 * (boundary_classifier[i-2] + boundary_classifier[i-1] +
        #            boundary_classifier[i] + boundary_classifier[i+1] + boundary_classifier[i+2])
        m = 1/5 * \
            (boundary_classifier[i-1] +
             boundary_classifier[i] + boundary_classifier[i+1])
        if m > 0.5:
            boundary_conditions['Neumann'].append(boundary_nodes[i])
        elif m < 0.5:
            boundary_conditions['Dirichlet'].append(boundary_nodes[i])
        else:
            if boundary_classifier[i] == 0:
                boundary_conditions['Dirichlet'].append(boundary_nodes[i])
            else:
                boundary_conditions['Neumann'].append(boundary_nodes[i])
    boundary_conditions['Dirichlet'] = np.unique(
        np.concatenate((boundary_conditions['Dirichlet'], upwind_nodes)))
    boundary_conditions['Neumann'] = [
        x for x in boundary_conditions['Neumann'] if x not in upwind_nodes]
    return boundary_conditions


def column_around_wind(nodes, boundary_nodes, wind_velocity):
    """
    Identify boundary nodes within a column aligned with the wind direction.

    Args:
        nodes (numpy.ndarray): Array of node coordinates.
        boundary_nodes (numpy.ndarray): Indices of boundary nodes.
        wind_velocity (numpy.ndarray): Wind velocity vector.

    Returns:
        numpy.ndarray: Indices of boundary nodes within the column.
    """
    Southampton = np.array([442365, 115483])
    m = wind_velocity[1]/wind_velocity[0]
    c = Southampton[1] - m*Southampton[0]
    s = 100000  # spacing between wind vector and column sides
    dc = abs(s / np.cos(np.arctan(m)))
    return np.where((nodes[boundary_nodes][:, 1] - m*nodes[boundary_nodes][:, 0] > c-dc) & (nodes[boundary_nodes][:, 1] - m*nodes[boundary_nodes][:, 0] < c+dc) & (nodes[boundary_nodes][:, 1] > 200000))[0]


def create_boundary_conditions(nodes, ID, boundary_nodes, wind_velocity, btype):
    """
    Generate boundary conditions based on the wind velocity and user-specified type.

    Args:
        nodes (numpy.ndarray): Array of node coordinates.
        ID (numpy.ndarray): Node index array for equations.
        boundary_nodes (numpy.ndarray): Indices of boundary nodes.
        wind_velocity (numpy.ndarray): Wind velocity vector.
        btype (int): Type of boundary condition to apply.

    Returns:
        tuple: A dictionary of boundary conditions and the updated ID array.
    """
    southern_border = np.where(nodes[boundary_nodes, 1] <= 200000)[0]
    upwind_nodes = upwind(nodes[boundary_nodes])
    column_nodes = column_around_wind(nodes, boundary_nodes, wind_velocity)

    boundary_types = in_and_out_flow_boundaries(
        nodes, boundary_nodes)
    # just upwind Direchlet
    if btype == 2:
        boundary_conditions = {
            "Dirichlet": upwind_nodes,
            "Neumann": [x for x in boundary_nodes if x not in upwind_nodes]
        }
    # Attempt to remove frequent changing in the boundary type which doesn't really make sense
    elif btype == 3:
        boundary_conditions = removing_irregularities_in_boundary(
            boundary_nodes, southern_border, boundary_types)
    # Just southern Direchlet
    elif btype == 4:
        boundary_conditions = {
            "Dirichlet": southern_border,
            "Neumann": [x for x in boundary_nodes if x not in southern_border]
        }
    # Inflow Dirichelt and upwind
    elif btype == 5:
        print('here')
        boundary_conditions = {
            "Dirichlet": np.unique(np.concatenate((boundary_types["Inflow"], upwind_nodes))),
            "Neumann": np.array([x for x in boundary_types["Outflow"] if x not in upwind_nodes])
        }
    # inflow Dirichlet and above 200000
    elif btype == 6:
        boundary_conditions = {
            "Dirichlet": np.unique(np.concatenate((boundary_types["Inflow"], southern_border))),
            "Neumann": np.array([x for x in boundary_types["Outflow"] if x not in southern_border])
        }
    # all Direchlet
    elif btype == 7:
        boundary_conditions = {
            "Dirichlet": boundary_nodes,
            "Neumann": []
        }
    # inflow outflow
    elif btype == 8:
        boundary_conditions = {
            "Dirichlet": boundary_types["Inflow"],
            "Neumann": boundary_types["Outflow"]
        }
    # Sets upwind of southampton nodes and inflow nodes excluding column nodes as Direchlet
    # Sets Outflow and column nodes excluding upwind as Neumann
    else:
        boundary_conditions = {
            "Dirichlet": np.unique(np.concatenate((np.array([x for x in boundary_types["Inflow"] if x not in column_nodes]), upwind_nodes))),
            "Neumann": np.array([x for x in np.unique(np.concatenate((boundary_types["Outflow"], column_nodes))) if x not in upwind_nodes])
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
    return boundary_conditions, ID


def convergence_test():
    """
    Perform a convergence test by comparing solutions across multiple grid resolutions.

    Args:
        None

    Plots:
        Generates and displays plots of convergence analysis.
    """
    res_dict = {1.25: '1_25',
                2.5: '2_5',
                5: '5',
                10: '10',
                20: '20',
                40: '40'}
    Psis = []
    psi_at_readings = []
    avg_psi_at_reading = []
    max_psi_at_reading = []
    for i, j in res_dict.items():
        Psi = np.load(
            'Finite_Element_code/Psi_for_different_grids/Time_dependent_Backwards Euler_dt10_T43200_D10000_m1_a1_f1_res'+str(i)+'_fr100_v10.npy')
        Psis.append(Psi.copy())
        nodes, IEN, boundary_nodes, ID = south_grid(j)
        tri = find_Reading(nodes, IEN)
        psi_values = Psi[:, tri]
        concentration = Psi_at_reading(psi_values, nodes[tri])
        psi_at_readings.append(concentration)
        # maybe make it concentration[concentration>0.01]
        avg_psi_at_reading.append(np.mean(concentration))
        max_psi_at_reading.append(np.max(concentration))
    avg_errors = [abs(val - avg_psi_at_reading[0])
                  for val in avg_psi_at_reading[1:]]
    max_errors = [abs(val - max_psi_at_reading[0])
                  for val in max_psi_at_reading[1:]]
    max_s = []
    avg_s = []
    for i in range(len(avg_psi_at_reading)-2):
        max_s.append(np.log2(abs(max_psi_at_reading[i+1] - max_psi_at_reading[i+2]) / abs(
            max_psi_at_reading[i] - max_psi_at_reading[i+1])))
        avg_s.append(np.log2(abs(avg_psi_at_reading[i+1] - avg_psi_at_reading[i+2]) / abs(
            avg_psi_at_reading[i] - avg_psi_at_reading[i+1])))

    h = np.array(list(res_dict.keys())[1:])*1000
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

    ax1.scatter((1/h[:-1])[::-1], avg_s[::-1])
    ax1.set_xticks((1/h[:-1])[::-1], 1 /
                   (np.array(list(res_dict.keys()))[1:-1])[::-1])

    ax2.scatter((1/h[:-1])[::-1], max_s[::-1])
    ax2.set_xticks((1/h[:-1])[::-1], 1 /
                   (np.array(list(res_dict.keys()))[1:-1])[::-1])
    fig.supylabel('Order of Convergence')
    fig.supxlabel('Grid Spacing')
    plt.show()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(9, 3))
    lobf_avg = np.poly1d(np.polyfit(h, avg_errors, 1))
    lobg_max = np.poly1d(np.polyfit(h, max_errors, 1))
    ax1.scatter(h, avg_errors, c='#0bb4ff', edgecolors='black')
    ax1.plot(h, lobf_avg(h), c='#0bb4ff',
             label=f'Order of Convergence:{np.round(np.polyfit(h, avg_errors, 1)[0], 4)}')
    ax2.scatter(h, max_errors, c='#50e991', edgecolors='black')
    ax2.plot(h, lobg_max(h), c='#50e991',
             label=f'Order of Convergence:{np.round(np.polyfit(h, max_errors, 1)[0], 4)}')
    axbox = ax1.get_position()
    fig.legend(loc='upper left', bbox_to_anchor=[
               axbox.x0, axbox.y0 + axbox.height])
    plt.show()
    print((np.polyfit(h, max_errors, 1)[0]))


def compute_matrices(nodes, IEN, ID, boundary_conditions, wind_velocity):
    """
    Assemble the global system matrices (M, K, A) and force vector (F) for the FEM.

    Args:
        nodes (numpy.ndarray): Array of node coordinates.
        IEN (numpy.ndarray): Element connectivity array.
        ID (numpy.ndarray): Node index array for equations.
        boundary_conditions (dict): Dictionary of boundary conditions.
        wind_velocity (numpy.ndarray): Wind velocity vector.

    Returns:
        tuple: Matrices (M, K, A) and force vector (F).
    """
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


def Backwards_Euler(M, K, A, F, ID, nodes, boundary_conditions, dt, T):
    """
    Solve the time-dependent PDE using the Backward Euler method.

    Args:
        M (scipy.sparse matrix): Mass matrix.
        K (scipy.sparse matrix): Stiffness matrix.
        A (scipy.sparse matrix): Advection matrix.
        F (numpy.ndarray): Force vector.
        ID (numpy.ndarray): Node index array for equations.
        nodes (numpy.ndarray): Array of node coordinates.
        boundary_conditions (dict): Boundary conditions dictionary.
        dt (float): Time step size.
        T (float): Total simulation time.

    Returns:
        numpy.ndarray: Psi solutions at each time step.
    """
    # Stability: None
    N_nodes = nodes.shape[0]
    N_equations = np.max(ID) + 1

    Psi_n = np.zeros(N_nodes)
    system_matrix = M + dt * (K - A)
    Psi_frames = []

    for t in np.arange(0, T, dt):
        if t < 28800:
            rhs = M @ Psi_n[ID >= 0] + dt * F
        else:
            rhs = M @ Psi_n[ID >= 0]
        Psi_interior = sp.linalg.spsolve(sp.csr_matrix(system_matrix), rhs)
        Psi_n[ID >= 0] = Psi_interior
        for n in boundary_conditions["Dirichlet"]:
            Psi_n[n] = 0
        Psi_frames.append(Psi_n.copy())

    return np.array(Psi_frames)


def Forward_Euler(M, K, A, F, ID, nodes, boundary_conditions, dt, T):
    """
    Solve the time-dependent PDE using the Forward Euler method.

    Args:
        M (scipy.sparse matrix): Mass matrix.
        K (scipy.sparse matrix): Stiffness matrix.
        A (scipy.sparse matrix): Advection matrix.
        F (numpy.ndarray): Force vector.
        ID (numpy.ndarray): Node index array for equations.
        nodes (numpy.ndarray): Array of node coordinates.
        boundary_conditions (dict): Boundary conditions dictionary.
        dt (float): Time step size.
        T (float): Total simulation time.

    Returns:
        numpy.ndarray: Psi solutions at each time step.
    """
    # Stability: dt <= min(dx^2/2D,dx/u)
    M = sp.csr_matrix(M)
    K = sp.csr_matrix(K)
    A = sp.csr_matrix(A)

    N_nodes = nodes.shape[0]
    Psi_n = np.zeros(N_nodes)  # Initialize Psi at t=0
    Psi_frames = []  # Store solutions at each time step

    for t in np.arange(0, T, dt):
        # Compute the right-hand side
        rhs = (K - A) @ Psi_n[ID >= 0] + F

        # Update Psi using Forward Euler
        Psi_interior = sp.linalg.spsolve(M, rhs)
        Psi_n[ID >= 0] += dt * Psi_interior

        # Apply Dirichlet boundary conditions
        for n in boundary_conditions["Dirichlet"]:
            Psi_n[n] = 0

        # Store the solution for visualization
        Psi_frames.append(Psi_n.copy())

    return np.array(Psi_frames)


def rk2(M, K, A, F, ID, nodes, boundary_conditions, dt, T):
    """
    Solve the time-dependent PDE using the Runge-Kutta 2 method.

    Args:
        M (scipy.sparse matrix): Mass matrix.
        K (scipy.sparse matrix): Stiffness matrix.
        A (scipy.sparse matrix): Advection matrix.
        F (numpy.ndarray): Force vector.
        ID (numpy.ndarray): Node index array for equations.
        nodes (numpy.ndarray): Array of node coordinates.
        boundary_conditions (dict): Boundary conditions dictionary.
        dt (float): Time step size.
        T (float): Total simulation time.

    Returns:
        numpy.ndarray: Psi solutions at each time step.
    """
    # Stability: dt <= min(dx^2/4D,dx/u)
    M = sp.csr_matrix(M)
    K = sp.csr_matrix(K)
    A = sp.csr_matrix(A)

    N_nodes = nodes.shape[0]
    Psi_n = np.zeros(N_nodes)
    Psi_frames = []

    for t in np.arange(0, T, dt):
        k1 = sp.linalg.spsolve(M, ((K-A) @ Psi_n[ID >= 0] + F))
        k2 = sp.linalg.spsolve(M, ((K-A) @ (Psi_n[ID >= 0] + dt/2 * k1) + F))
        Psi_n[ID >= 0] += dt * k2

        for n in boundary_conditions["Dirichlet"]:
            Psi_n[n] = 0

        Psi_frames.append(Psi_n.copy())

    return np.array(Psi_frames)


def Crank_Nicholson(M, K, A, F, ID, nodes, boundary_conditions, dt, T):
    """
    Solve the time-dependent PDE using the Crank-Nicholson method.

    Args:
        M (scipy.sparse matrix): Mass matrix.
        K (scipy.sparse matrix): Stiffness matrix.
        A (scipy.sparse matrix): Advection matrix.
        F (numpy.ndarray): Force vector.
        ID (numpy.ndarray): Node index array for equations.
        nodes (numpy.ndarray): Array of node coordinates.
        boundary_conditions (dict): Boundary conditions dictionary.
        dt (float): Time step size.
        T (float): Total simulation time.

    Returns:
        numpy.ndarray: Psi solutions at each time step.
    """
    # Stability: None
    # Needs small diffusion
    M = sp.csr_matrix(M)
    K = sp.csr_matrix(K)
    A = sp.csr_matrix(A)

    N_nodes = nodes.shape[0]
    Psi_n = np.zeros(N_nodes)
    Psi_frames = []

    lhs = M - dt/2 * (K-A)
    part_rhs = M + dt/2 * (K-A)

    for t in np.arange(0, T, dt):
        rhs = part_rhs @ Psi_n[ID >= 0] + dt * F
        Psi_interior = sp.linalg.spsolve(lhs, rhs)
        Psi_n[ID >= 0] = Psi_interior
        for n in boundary_conditions["Dirichlet"]:
            Psi_n[n] = 0
        Psi_frames.append(Psi_n.copy())

    return np.array(Psi_frames)


def plot_results(nodes, IEN, boundary_conditions, boundary_nodes, Psi_A):
    """
    Plot the results of the simulation, highlighting boundary conditions and key locations.

    Args:
        nodes (numpy.ndarray): Array of node coordinates.
        IEN (numpy.ndarray): Element connectivity array.
        boundary_conditions (dict): Dictionary of Dirichlet and Neumann boundary conditions.
        boundary_nodes (numpy.ndarray): Indices of boundary nodes.
        Psi_A (numpy.ndarray): Values of the solution at each node.

    Returns:
        None: Displays the plot of the simulation results.
    """
    plt.figure(figsize=(5, 5))
    plt.tripcolor(nodes[:, 0], nodes[:, 1], Psi_A, triangles=IEN)
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
    # plt.colorbar()
    plt.legend('upper left')
    plt.show()


def plot_frames(nodes, IEN, boundary_conditions, boundary_nodes, Psi_frames, frames, file_name):
    """
    Plot and save specific frames from the simulation results.

    Args:
        nodes (numpy.ndarray): Array of node coordinates.
        IEN (numpy.ndarray): Element connectivity array.
        boundary_conditions (dict): Dictionary of Dirichlet and Neumann boundary conditions.
        boundary_nodes (numpy.ndarray): Indices of boundary nodes.
        Psi_frames (numpy.ndarray): Array of solution frames from the simulation.
        frames (list): Specific frames to plot.
        file_name (str): Filename prefix for saving the plots.

    Returns:
        None: Saves the plots as PDF files.
    """
    for i, Psi in enumerate(Psi_frames):
        plt.figure(figsize=(5, 5))
        plt.tripcolor(nodes[:, 0], nodes[:, 1], Psi,
                      triangles=IEN, shading='gouraud')
        plt.scatter(nodes[boundary_conditions["Dirichlet"], 0],
                    nodes[boundary_conditions["Dirichlet"], 1],
                    color='orange', label='Dirichlet Boundary Nodes', s=1)
        plt.scatter(nodes[boundary_conditions["Neumann"], 0],
                    nodes[boundary_conditions["Neumann"], 1],
                    color='lightblue', label='Neumann Boundary Nodes', s=1)
        plt.scatter([442365], [115483], marker='o',
                    facecolor='None', edgecolor='black', label='Southampton')
        plt.scatter([473993], [171625], marker='o',
                    facecolor='None', edgecolor='black', label='Reading')

        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        plt.savefig(
            f'Plots2/Frame{frames[i]}{file_name}.pdf', bbox_inches="tight")
        print(f'file saved as Frame{frames[i]}')


def create_gif(Psi_frames, nodes, IEN, filename, dt, frame_rate, boundary_conditions):
    """
    Create and save a GIF visualizing the time evolution of the simulation results.

    Args:
        Psi_frames (numpy.ndarray): Array of solution frames from the simulation.
        nodes (numpy.ndarray): Array of node coordinates.
        IEN (numpy.ndarray): Element connectivity array.
        filename (str): Filename for saving the GIF.
        dt (float): Time step size.
        frame_rate (int): Frame rate for the GIF.
        boundary_conditions (dict): Dictionary of Dirichlet and Neumann boundary conditions.

    Returns:
        None: Saves the GIF to the specified file.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    triang = Triangulation(nodes[:, 0], nodes[:, 1], IEN)
    contour = ax.tripcolor(
        triang, Psi_frames[0], shading='flat', cmap='viridis', vmin=0, vmax=1)
    # cbar = plt.colorbar(contour, ax=ax)
    # cbar.set_label("Normalized $\\Psi$")
    # cbar.ax.tick_params(labelsize=8)

    # plot the perp to wind line
    # wind_direction = wind_velocity(1, 'Crank-Nicholson')
    # perp_to_wind = np.array([-wind_direction[1], wind_direction[0]])
    # Southampton = np.array([442365, 115483])
    # m = perp_to_wind[1]/perp_to_wind[0]
    # c = Southampton[1] - m*Southampton[0]
    # x = np.arange(400000, 500000)
    # y = m * x + c
    ###

    images = []
    for i, Psi in enumerate(Psi_frames):
        if i % frame_rate == 0:
            ax.clear()
            contour = ax.tripcolor(triang, Psi, shading='flat',
                                   cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f"Time Evolution: t = {i * dt:.1f} seconds")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.scatter([442365], [115483], marker='o',
                       facecolor='None', edgecolor='black', label='Southampton')
            ax.scatter([473993], [171625], marker='o',
                       facecolor='None', edgecolor='black', label='Reading')
            ax.scatter(nodes[boundary_conditions["Dirichlet"], 0],
                       nodes[boundary_conditions["Dirichlet"], 1],
                       color='orange', label='Dirichlet Boundary Nodes', marker='.')
            ax.scatter(nodes[boundary_conditions["Neumann"], 0],
                       nodes[boundary_conditions["Neumann"], 1],
                       color='lightblue', label='Neumann Boundary Nodes', marker='.')
            # ax.plot(x, y, c='k') #plot the perp to wind line
            # Render the figure and convert it to an image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(
                fig.canvas.get_width_height()[::-1] + (4,)
            )
            # Remove the alpha channel
            image_rgb = Image.fromarray(image[:, :, :3])
            images.append(image_rgb)

    # Save as GIF
    images[0].save(
        f'Plots2/{filename}.gif', save_all=True, append_images=images[1:], duration=100, loop=0
    )
    print(f"GIF saved as {filename}")


def non_dimensionalise(M, K, A, F, L, windspeed):
    """
    Non-dimensionalize the FEM matrices and force vector.

    Args:
        M (scipy.sparse matrix): Mass matrix.
        K (scipy.sparse matrix): Stiffness matrix.
        A (scipy.sparse matrix): Advection matrix.
        F (numpy.ndarray): Force vector.
        L (float): Characteristic length scale.
        windspeed (float): Characteristic wind velocity.

    Returns:
        tuple: Non-dimensionalized matrices (M, K, A) and force vector (F).
    """
    M = sp.csr_matrix(M)
    A = sp.csr_matrix(A)
    K = sp.csr_matrix(K)
    return M / L**2, K, A / (windspeed * L), F / L**2


def scaling_matrices(M, K, A, F, L, windspeed):
    """
    Scale the FEM matrices and force vector to match expected magnitudes.

    Args:
        M (scipy.sparse matrix): Mass matrix.
        K (scipy.sparse matrix): Stiffness matrix.
        A (scipy.sparse matrix): Advection matrix.
        F (numpy.ndarray): Force vector.
        L (float): Characteristic length scale.
        windspeed (float): Characteristic wind velocity.

    Returns:
        tuple: Scaled matrices (M, K, A) and force vector (F).
    """
    M = sp.csr_matrix(M)
    A = sp.csr_matrix(A)
    K = sp.csr_matrix(K)
    mean_m = np.absolute(M[M != 0]).mean()
    mean_k = np.absolute(K[K != 0]).mean()
    mean_a = np.absolute(A[A != 0]).mean()
    mean_f = np.absolute(F[F != 0]).mean()
    scale_m = L**2/mean_m
    scale_k = 1
    scale_a = windspeed * L / mean_a
    scale_f = L**2 / mean_f
    return M * scale_m, K * scale_k, A * scale_a, F * scale_f


def make_gif_from_scratch(btype):
    """
    Create and save a simulation GIF by running the solver from scratch.

    Args:
        btype (int): Boundary condition type to use.

    Returns:
        None: Generates and saves a GIF of the simulation results.
    """
    windspeed = 10
    D = 0
    m = 1
    f = 1
    a = 1
    frame_rate = 100
    res = 5
    res_dict = {1.25: '1_25',
                2.5: '2_5',
                5: '5',
                10: '10',
                20: '20',
                40: '40'}
    model = 4
    model_name = {1: 'Backwards Euler',
                  2: 'Forwards Euler',
                  3: 'Runge-Kutta 2',
                  4: 'Crank-Nicholson'}
    models = {'Backwards Euler': Backwards_Euler,
              'Forwards Euler': Forward_Euler,
              'Runge-Kutta 2': rk2,
              'Crank-Nicholson': Crank_Nicholson}
    nodes, IEN, boundary_nodes, ID = south_grid(res_dict[res])
    boundary_conditions, ID2 = create_boundary_conditions(
        nodes, ID, boundary_nodes, wind_velocity(windspeed, model_name[model]), model_name[model], btype)

    M, K, A, F = compute_matrices(
        nodes, IEN, ID2, boundary_conditions, wind_velocity(windspeed, model_name[model]))

    dt = 10
    T = 43200  # 12hours
    T = 10000
    # # Attempt to non-dimensionalise
    # # S_0 = 1
    # # L = res*1000
    # # U = windspeed
    # # M_nd = M / L**2
    # # K_nd = K * L**2
    # # A_nd = A * L / U
    # # F_nd = F * L**2 / S_0
    # # print('K_nd =', np.absolute(K_nd[K_nd != 0]).mean())
    # # print('A_nd =', np.absolute(A_nd[A_nd != 0]).mean())
    # # print('M_nd =', np.absolute(M_nd[M_nd != 0]).mean())
    # # print('F_nd =', np.absolute(F_nd[F_nd != 0]).mean())
    # # Psi_frames = models[model_name[model]](
    # #     M_nd, K_nd, A_nd, F_nd, ID2, nodes, boundary_conditions, dt, T)

    L = res*1000

    # # M, K, A, F = scaling_matrices(M, K, A, F, L, windspeed)

    print('L =', L, 'u =', windspeed)
    print('D * K =', np.round(np.absolute(D *
          K[K != 0]).mean(), decimals=5), 'D * K ~ L^2', L**2)
    print('A =', np.round(np.absolute(
        A[A != 0]).mean(), 5), 'A ~ u * L', windspeed*L)
    print('m * M =', np.round(np.absolute(m *
          M[M != 0]).mean(), 5), 'M ~ L^2', L**2)
    print('f * F =', np.round(np.absolute(f *
          F[F != 0]).mean(), 5), 'F ~ L^2', L**2)

    Psi_frames = models[model_name[model]](
        m * M, D*K, a * A, f * F, ID2, nodes, boundary_conditions, dt, T)

    Psi_frames, psi_min, psi_max = normalize_psi(Psi_frames)
    tri = find_Reading(nodes, IEN)
    psi_values = Psi_frames[:, tri]
    file_name = f"{model_name[model]}_dt{dt}_T{T}_" \
        + f"D{D}_m{m}_a{a}_f{f}_res{res}_fr{frame_rate}_v{windspeed}"

    # Psi_at_reading_plot(psi_values, nodes[tri], dt, T, file_name)

    file_name = "Test5"
    create_gif(Psi_frames, nodes, IEN,
               file_name, dt, frame_rate, boundary_conditions)
    print('Done')


def save_psi(res):
    """
    Save the simulation results (Psi) for a specified grid resolution.

    Args:
        res (float): Grid resolution to use for the simulation.

    Returns:
        None: Saves the Psi array to a file.
    """
    windspeed = 10
    D = 10000
    m = 1
    f = 1
    a = 1
    frame_rate = 100
    btype = 1
    res_dict = {1.25: '1_25',
                2.5: '2_5',
                5: '5',
                10: '10',
                20: '20',
                40: '40'}
    model = 1
    model_name = {1: 'Backwards Euler',
                  2: 'Forwards Euler',
                  3: 'Runge-Kutta 2',
                  4: 'Crank-Nicholson'}
    models = {'Backwards Euler': Backwards_Euler,
              'Forwards Euler': Forward_Euler,
              'Runge-Kutta 2': rk2,
              'Crank-Nicholson': Crank_Nicholson}
    nodes, IEN, boundary_nodes, ID = south_grid(res_dict[res])
    boundary_conditions, ID2 = create_boundary_conditions(
        nodes, ID, boundary_nodes, wind_velocity(windspeed, model_name[model]), btype)
    M, K, A, F = compute_matrices(
        nodes, IEN, ID2, boundary_conditions, wind_velocity(windspeed, model_name[model]))

    dt = 10
    T = 43200  # 12hours

    Psi_frames = models[model_name[model]](
        m * M, D*K, a * A, f * F, ID2, nodes, boundary_conditions, dt, T)

    Psi_frames, psi_min, psi_max = normalize_psi(Psi_frames)

    np.save(f'Finite_Element_code/Psi_for_different_grids/Time_dependent_{model_name[model]}_dt{dt}_T{T}_'
            + f'D{D}_m{m}_a{a}_f{f}_res{res}_fr{frame_rate}_v{windspeed}.npy', Psi_frames)
    print('Done')


def make_gif_from_file():
    """
    Create and save a simulation GIF using precomputed Psi frames from a file.

    Args:
        None

    Returns:
        None: Generates and saves a GIF of the simulation results.
    """
    windspeed = 10
    D = 10000
    m = 1
    f = 1
    a = 1
    frame_rate = 100
    res = 1.25
    btype = 1
    res_dict = {1.25: '1_25',
                2.5: '2_5',
                5: '5',
                10: '10',
                20: '20',
                40: '40'}
    model = 1
    model_name = {1: 'Backwards Euler',
                  2: 'Forwards Euler',
                  3: 'Runge-Kutta 2',
                  4: 'Crank-Nicholson'}
    models = {'Backwards Euler': Backwards_Euler,
              'Forwards Euler': Forward_Euler,
              'Runge-Kutta 2': rk2,
              'Crank-Nicholson': Crank_Nicholson}
    nodes, IEN, boundary_nodes, ID = south_grid(res_dict[res])
    boundary_conditions, ID2 = create_boundary_conditions(
        nodes, ID, boundary_nodes, wind_velocity(windspeed, model_name[model]), btype)

    dt = 10
    T = 43200  # 12hours

    Psi_frames = np.load(
        'Finite_Element_code/Psi_for_different_grids/Time_dependent_Backwards Euler_dt10_T43200_D10000_m1_a1_f1_res1.25_fr100_v10.npy')

    tri = find_Reading(nodes, IEN)
    psi_values = Psi_frames[:, tri]
    Psi_at_reading_plot(psi_values, nodes[tri], dt, T, D, m, a, f,
                        res, frame_rate, windspeed, model_name[model])
    # Generate and save GIF
    file_name = f"Plots2/Time_dependent_{model_name[model]}_dt{dt}_T{T}_" \
        + f"D{D}_m{m}_a{a}_f{f}_res{res}_fr{frame_rate}_v{windspeed}.gif"
    file_name = "Plots2/Test2.gif"
    create_gif(Psi_frames, nodes, IEN,
               file_name, dt, frame_rate, boundary_conditions)
    print('Finished')


def plot_specific_frames_from_scratch(frames, btype, concentration_at_reading=False):
    """
    Plot specific frames from a simulation and optionally plot the concentration over Reading.

    Args:
        frames (list): Specific time frames to plot.
        btype (int): Boundary condition type to use.
        concentration_at_reading (bool, optional): Whether to plot concentration over Reading.

    Returns:
        None: Saves the plots as PDF files.
    """
    windspeed = 10
    # D = backwards 10000, rest 0
    D = 1000
    m = 1
    f = 1
    a = 1
    res = 5
    res_dict = {1.25: '1_25',
                2.5: '2_5',
                5: '5',
                10: '10',
                20: '20',
                40: '40'}
    model = 3
    model_name = {1: 'Backwards Euler',
                  2: 'Forwards Euler',
                  3: 'Runge-Kutta 2',
                  4: 'Crank-Nicholson'}
    models = {'Backwards Euler': Backwards_Euler,
              'Forwards Euler': Forward_Euler,
              'Runge-Kutta 2': rk2,
              'Crank-Nicholson': Crank_Nicholson}
    nodes, IEN, boundary_nodes, ID = south_grid(res_dict[res])
    boundary_conditions, ID2 = create_boundary_conditions(
        nodes, ID, boundary_nodes, wind_velocity(windspeed, model_name[model]), btype)
    M, K, A, F = compute_matrices(
        nodes, IEN, ID2, boundary_conditions, wind_velocity(windspeed, model_name[model]))

    dt = 0.1
    T = max(frames)+1

    Psi_frames = models[model_name[model]](
        m * M, D*K, a * A, f * F, ID2, nodes, boundary_conditions, dt, T)

    Psi_frames, psi_min, psi_max = normalize_psi(Psi_frames)
    file_name = f"_{model_name[model]}_dt{dt}_T{T}_" \
        + f"D{D}_m{m}_a{a}_f{f}_res{res}_v{windspeed}_b{btype}"

    if concentration_at_reading == True:
        tri = find_Reading(nodes, IEN)
        psi_values = Psi_frames[:, tri]
        Psi_at_reading_plot(psi_values, nodes[tri], dt, T, file_name)

    adjusted_frames = [int(frame//dt-1) for frame in frames]
    plot_frames(nodes, IEN, boundary_conditions,
                boundary_nodes, Psi_frames[adjusted_frames], frames, file_name)


start = time.time()

frames = [100]
plot_specific_frames_from_scratch(
    frames, btype=1)
# make_gif_from_scratch(btype=5)

end = time.time()
print(f'Completion time {end-start}')
