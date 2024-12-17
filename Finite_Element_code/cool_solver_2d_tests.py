import Finite_Element_code.solver_2d as s2d
import numpy as np


def test_local2globalCoords():

    default = {
        "xe": np.array([[0, 1, 0],
                        [0, 0, 1]]),
        "ans": np.array([0.5, 0.5])
    }

    translated = {
        "xe": np.array([[1, 2, 1],
                        [0, 0, 1]]),
        "ans": np.array([1.5, 0.5])
    }

    scaled = {
        "xe": np.array([[0, 2, 0],
                        [0, 0, 2]]),
        "ans": np.array([1, 1])
    }

    rotated = {
        "xe": np.array([[1, 0, 1],
                        [1, 1, 0]]),
        "ans": np.array([0.5, 0.5])
    }

    xi = np.array([0.5, 0.5])

    for t in [default, translated, scaled, rotated]:
        assert np.allclose(s2d.global_x_of_xi(xi, t["xe"].T),
                           t["ans"]), f"element\n {t['xe']} is broken"


test_local2globalCoords()


def test_jacobian():

    default = {
        "xe": np.array([[0, 1, 0],
                        [0, 0, 1]]),
        "ans": np.array([[1, 0],
                         [0, 1]])
    }

    translated = {
        "xe": np.array([[1, 2, 1],
                        [0, 0, 1]]),
        "ans": np.array([[1, 0],
                         [0, 1]])
    }

    scaled = {
        "xe": np.array([[0, 2, 0],
                        [0, 0, 2]]),
        "ans": np.array([[2, 0],
                         [0, 2]])
    }

    rotated = {
        "xe": np.array([[1, 0, 1],
                        [1, 1, 0]]),
        "ans": np.array([[-1, 0],
                         [0, -1]])
    }

    for t in [default, translated, scaled, rotated]:
        assert np.allclose(s2d.jacobian(t["xe"].T),
                           t["ans"]), f"element\n {t['xe']} is broken"


test_jacobian()


def test_globalShapeFunctionDerivatives():

    default = {
        "xe": np.array([[0, 1, 0],
                        [0, 0, 1]]),
        "ans": np.array([[-1, 1, 0],
                         [-1, 0, 1]])
    }

    translated = {
        "xe": np.array([[1, 2, 1],
                        [0, 0, 1]]),
        "ans": np.array([[-1, 1, 0],
                         [-1, 0, 1]])
    }

    scaled = {
        "xe": np.array([[0, 2, 0],
                        [0, 0, 2]]),
        "ans": np.array([[-0.5, 0.5, 0],
                         [-0.5, 0, 0.5]])
    }

    rotated = {
        "xe": np.array([[1, 0, 1],
                        [1, 1, 0]]),
        "ans": np.array([[1, -1, 0],
                         [1, 0, -1]])
    }

    for t in [default, translated, scaled, rotated]:
        assert np.allclose(s2d.global_derivative_of_shape_functions(t["xe"].T),
                           t["ans"].T), f"element\n {t['xe']} is broken"


test_globalShapeFunctionDerivatives()


def test_localQuadtrature():

    constant = {
        "psi": lambda x: 1,
        "ans": 0.5
    }

    linearx = {
        "psi": lambda x: 6*x[0],
        "ans": 1,
    }

    lineary = {
        "psi": lambda x: x[1],
        "ans": 1/6,
    }

    product = {
        "psi": lambda x: x[0]*x[1],
        "ans": 1/24,
    }
    for t in [constant, linearx, lineary, product]:
        assert np.allclose(s2d.quadrature_over_reference_triangle(t["psi"]),
                           t["ans"]), f"function with answer\n {t['ans']} is broken"


def test_globalQuadrature():

    translated_linear = {
        "xe": np.array([[1, 2, 1],
                        [0, 0, 1]]),
        "phi": lambda x: 3*x[0],
        "ans": 2,
    }

    translated_product = {
        "xe": np.array([[1, 2, 1],
                        [0, 0, 1]]),
        "phi": lambda x: x[0]*x[1],
        "ans": 5/24,
    }

    scaled_linear = {
        "xe": np.array([[0, 2, 0],
                        [0, 0, 2]]),
        "phi": lambda x: 3*x[0],
        "ans": 4,
    }

    scaled_product = {
        "xe": np.array([[0, 2, 0],
                        [0, 0, 2]]),
        "phi": lambda x: x[0]*x[1],
        "ans": 2/3,
    }

    rotated_linear = {
        "xe": np.array([[1, 0, 1],
                        [1, 1, 0]]),
        "phi": lambda x: 3*x[0],
        "ans": 1,
    }

    rotated_product = {
        "xe": np.array([[1, 0, 1],
                        [1, 1, 0]]),
        "phi": lambda x: x[0]*x[1],
        "ans": 5/24,
    }

    for t in [translated_linear, translated_product, scaled_linear, scaled_product,
              rotated_linear, rotated_product]:
        assert np.allclose(s2d.quadrature_over_element(t["xe"].T, t["phi"]),
                           t["ans"]), f"element\n {t['xe']} with answer\n {t['ans']} is broken"


def test_diffusion_stiffness():

    default = {
        "xe": np.array([[0, 1, 0],
                        [0, 0, 1]]),
        "ans": np.array([[1, -0.5, -0.5],
                         [-0.5, 0.5, 0],
                         [-0.5, 0, 0.5]])
    }

    translated = {
        "xe": np.array([[1, 2, 1],
                        [0, 0, 1]]),
        "ans": np.array([[1, -0.5, -0.5],
                         [-0.5, 0.5, 0],
                         [-0.5, 0, 0.5]])
    }

    scaled = {
        "xe": np.array([[0, 2, 0],
                        [0, 0, 2]]),
        "ans": np.array([[1, -0.5, -0.5],
                         [-0.5, 0.5, 0],
                         [-0.5, 0, 0.5]])
    }
    rotated = {
        "xe": np.array([[1, 0, 1],
                        [1, 1, 0]]),
        "ans": np.array([[1, -0.5, -0.5],
                         [-0.5, 0.5, 0],
                         [-0.5, 0, 0.5]])
    }

    for t in [default, translated, scaled, rotated]:
        assert np.allclose(s2d.stiffness_matrix_for_element(t["xe"].T),
                           t["ans"]), f"element\n {t['xe']} is broken"


def test_advection_stiffness():

    default = {
        "xe": np.array([[0, 1, 0],
                        [0, 0, 1]]),
        "ans": 1/6 * np.array([[-1, -1, -1],
                               [0,  0,  0],
                               [1,  1,  1]])
    }

    translated = {
        "xe": np.array([[1, 2, 1],
                        [0, 0, 1]]),
        "ans": 1/6 * np.array([[-1, -1, -1],
                               [0,  0,  0],
                               [1,  1,  1]])
    }

    scaled = {
        "xe": np.array([[0, 2, 0],
                        [0, 0, 2]]),
        "ans": 1/3 * np.array([[-1, -1, -1],
                               [0,  0,  0],
                               [1,  1,  1]])
    }
    rotated = {
        "xe": np.array([[1, 0, 1],
                        [1, 1, 0]]),
        "ans": 1/6 * np.array([[1,  1,  1],
                               [0,  0,  0],
                               [-1, -1, -1]])
    }

    for t in [default, translated, scaled, rotated]:
        assert np.allclose(s2d.advection_matrix_for_element(global_node_coords=t["xe"].T, y_wind_velocity=1),
                           t["ans"]), f"element\n {t['xe']} is broken"


def test_force():

    default_const = {
        "xe": np.array([[0, 1, 0],
                        [0, 0, 1]]),
        "S": lambda x: 1,
        "ans": 1/6 * np.array([1, 1, 1]),
    }

    default_linear = {
        "xe": np.array([[0, 1, 0],
                        [0, 0, 1]]),
        "S": lambda x: x[0],
        "ans": 1/24 * np.array([1, 2, 1])
    }

    translated_const = {
        "xe": np.array([[1, 2, 1],
                        [0, 0, 1]]),
        "S": lambda x: 1,
        "ans": 1/6 * np.array([1, 1, 1]),
    }

    translated_linear = {
        "xe": np.array([[1, 2, 1],
                        [0, 0, 1]]),
        "S": lambda x: x[1],
        "ans": 1/24 * np.array([1, 1, 2])
    }

    scaled_const = {
        "xe": np.array([[0, 2, 0],
                        [0, 0, 2]]),
        "S": lambda x: 1,
        "ans": 2/3 * np.array([1, 1, 1]),
    }

    scaled_linear = {
        "xe": np.array([[0, 2, 0],
                        [0, 0, 2]]),
        "S": lambda x: x[0],
        "ans": 1/3 * np.array([1, 2, 1]),
    }

    rotated_const = {
        "xe": np.array([[1, 0, 1],
                        [1, 1, 0]]),
        "S": lambda x: 1,
        "ans": 1/6 * np.array([1, 1, 1]),
    }

    rotated_linear = {
        "xe": np.array([[1, 0, 1],
                        [1, 1, 0]]),
        "S": lambda x: x[1],
        "ans": 1/4 * np.array([1/2, 1/2, 1/3]),
    }

    for t in [default_const, default_linear, translated_const, translated_linear,
              scaled_const, scaled_linear, rotated_const, rotated_linear]:
        assert np.allclose(s2d.force_vector_for_element(t["xe"].T, t["S"]),
                           t["ans"]), f"element\n {t['xe']} with answer\n {t['ans']} is broken"


print('Done')
