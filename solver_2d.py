import numpy as np


def ref_shape_functions(xi):
    """
    Compute the reference shape functions for a triangle.

    Args:
        xi (ndarray): Array of local coordinates in the reference triangle.

    Returns:
        ndarray: Array of shape functions evaluated at the given coordinates.
    """
    xi = np.atleast_2d(xi)
    return np.array([1-xi[:, 0]-xi[:, 1], xi[:, 0], xi[:, 1]]).T


def global_x_of_xi(xi, global_node_coords):
    """
    Maps coordinates from the reference triangle to global coordinates.

    Args:
        xi (ndarray): Array of local coordinates in the reference triangle.
        global_node_coords (ndarray): Coordinates of the triangle's nodes in the global space.

    Returns:
        ndarray: Global coordinates corresponding to the given local coordinates.
    """
    N = ref_shape_functions(xi)
    return np.dot(N, global_node_coords)


def derivative_of_ref_shape_functions():
    """
    Compute the derivatives of the reference shape functions with respect to the reference coordinates.

    Returns:
        ndarray: Derivatives of the reference shape functions.
    """
    return np.array([[-1, -1], [1, 0], [0, 1]])


def jacobian(node_coords):
    """
    Compute the Jacobian matrix for a triangle element. 
    This is the transpose of what it should be because that worked... maybe because of clockwise vs anticlockwise labelling

    Args:
        node_coords (ndarray): Global coordinates of the triangle's nodes.

    Returns:
        ndarray: Jacobian matrix of the element.
    """
    dN_dxi = derivative_of_ref_shape_functions()
    # return np.array(node_coords).T @ dN_dxi # correct jacobian
    return dN_dxi.T @ np.array(node_coords)  # what works


def det_jacobian(J):
    """
    Compute the determinant of the Jacobian matrix.

    Args:
        J (ndarray): Jacobian matrix.

    Returns:
        float: Determinant of the Jacobian matrix.
    """
    return J[0][0] * J[1][1] - J[0][1] * J[1][0]


def inverse_jacobian(J):
    """
    Compute the inverse of the Jacobian matrix.

    Args:
        J (ndarray): Jacobian matrix.

    Returns:
        ndarray: Inverse of the Jacobian matrix.
    """
    return np.array([[J[1][1], -J[0][1]], [-J[1][0], J[0][0]]])/det_jacobian(J)


def global_derivative_of_shape_functions(global_node_coords):
    """
    Compute the derivatives of the shape functions in global coordinates.

    Args:
        global_node_coords (ndarray): Global coordinates of the triangle's nodes.

    Returns:
        ndarray: Global derivatives of the shape functions.
    """
    dN_dxi = derivative_of_ref_shape_functions()
    J = jacobian(global_node_coords)
    J_inv = inverse_jacobian(J)
    return dN_dxi @ J_inv.T


def quadrature_over_reference_triangle(psi):
    """
    Perform numerical quadrature over the reference triangle. 
    I haven't checked this as I don't use it 

    Args:
        psi (function): Function to integrate over the reference triangle.

    Returns:
        float: Approximated integral value over the reference triangle.
    """
    ref_xi = [[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]]
    ref_tri_nodes = [[0, 0], [1, 0], [0, 1]]
    x = global_x_of_xi(ref_xi, ref_tri_nodes)
    return (1/6)*sum(psi(x))


def quadrature_over_element(global_node_coords, psi):
    """
    Perform numerical quadrature over a triangle element in global coordinates.
    I also haven't checked this function as it isn't used.

    Args:
        global_node_coords (ndarray): Global coordinates of the triangle's nodes.
        psi (function): Function to integrate over the element.

    Returns:
        float: Approximated integral value over the triangle element.
    """
    ref_xi = [[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]]
    xs = global_x_of_xi(ref_xi, global_node_coords)
    detJ = det_jacobian(jacobian(global_node_coords))
    return (1/6)*np.sum(psi(xs))*detJ


def stiffness_matrix_for_element(global_node_coords):
    """
    Compute the diffusion stiffness matrix for an element.

    Args:
        global_node_coords (ndarray): Global coordinates of the triangle's nodes.

    Returns:
        ndarray: Stiffness matrix of the element.
    """
    dN_dxi = derivative_of_ref_shape_functions()
    J = jacobian(global_node_coords)
    detJ = det_jacobian(J)
    J_inv = inverse_jacobian(J)
    # Shape: (2, 3) constant for all node points
    dN_dxdy = dN_dxi @ J_inv.T
    # summing over 3 time of the same and then doing 1/6 is just 1/2 it once
    return 1/2 * abs(detJ) * (dN_dxdy @ dN_dxdy.T)


def force_vector_for_element(global_node_coords, source_func):
    """
    Compute the force vector for an element.

    Args:
        global_node_coords (ndarray): Global coordinates of the triangle's nodes.
        source_func (function): Source term function defined in global coordinates.

    Returns:
        ndarray: Force vector for the element.
    """
    ref_xi = [[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]]
    N = ref_shape_functions(ref_xi)
    xys = global_x_of_xi(ref_xi, global_node_coords)
    detJ = det_jacobian(jacobian(global_node_coords))
    S_values = np.array([source_func(xy) for xy in xys])
    # print('xys', xys)
    # print('S_values', S_values)
    return np.reshape(1/6 * (N @ S_values) * abs(detJ), (3, 1))
    # return np.sum(1/6*S_values * N, axis=0)*detJ


# print(force_vector_for_element(
#     [[1, 1], [0, 1], [1, 0]], lambda x: x[0] + x[1]))


# def advection_matrix_for_element(global_node_coords, wind_velocity):
#     ref_xi = np.array([[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]])  # (3, 2)
#     N = ref_shape_functions(ref_xi)  # (3 quadrature points, 3 shape functions)
#     # print(f'N: {N}')
#     dN_dxi = derivative_of_ref_shape_functions()  # (3, 2) fixed
#     J = jacobian(global_node_coords)  # (2, 2)
#     detJ = det_jacobian(J)
#     J_inv = inverse_jacobian(J)  # (2, 2)
#     dN_dxdy = dN_dxi @ J_inv.T
#     dN_dx, dN_dy = dN_dxdy[:, 0], dN_dxdy[:, 1]
#     # print(f'dN_dxdy: {dN_dxdy}')
#     # print(f'dN_dy: {dN_dy}')
#     # dN = np.dot(wind_velocity,dN_dx)
#     # return 1/6 * abs(detJ) * np.sum(np.array([np.outer(N_xi, dN_dy) for N_xi in N]), axis=0)
#     advection_matrix_x = 1/6 * wind_velocity[0] * abs(detJ) * np.sum(
#         np.array([np.outer(N_xi, dN_dx) for N_xi in N]), axis=0)
#     advection_matrix_y = 1/6 * wind_velocity[1] * abs(detJ) * np.sum(
#         np.array([np.outer(N_xi, dN_dy) for N_xi in N]), axis=0)
#     return advection_matrix_x + advection_matrix_y

def advection_matrix_for_element(global_node_coords, wind_velocity):
    """
    Compute the advection matrix for an element.

    Args:
        global_node_coords (ndarray): Global coordinates of the triangle's nodes.
        wind_velocity (ndarray): Wind velocity vector in global coordinates.

    Returns:
        ndarray: Advection matrix for the element.
    """
    ref_xi = np.array([[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]])
    N = ref_shape_functions(ref_xi)
    dN_dxi = derivative_of_ref_shape_functions()
    J = jacobian(global_node_coords)
    detJ = det_jacobian(J)
    J_inv = inverse_jacobian(J)
    dN_dxdy = dN_dxi @ J_inv.T
    dN_dx, dN_dy = dN_dxdy[:, 0], dN_dxdy[:, 1]
    v_dot_gradN = wind_velocity[0] * dN_dx + wind_velocity[1] * dN_dy
    advection_matrix = 1/6 * detJ * \
        np.sum(np.array([np.outer(N_xi, v_dot_gradN) for N_xi in N]), axis=0)
    return advection_matrix


def mass_matrix_for_element(global_node_coords):
    """
    Compute the mass matrix for an element.

    Args:
        global_node_coords (ndarray): Global coordinates of the triangle's nodes.

    Returns:
        ndarray: Mass matrix of the element.
    """
    ref_xi = np.array([[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]])
    N = ref_shape_functions(ref_xi)
    detJ = det_jacobian(jacobian(global_node_coords))
    return 1/6 * abs(detJ) * np.sum(np.array([np.outer(N_xi, N_xi) for N_xi in N]), axis=0)
    # return 1/6 * detJ * N.T @ N
    # need to test here
