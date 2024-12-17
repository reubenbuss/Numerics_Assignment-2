import matplotlib.pyplot as plt
import numpy as np
def generate_2d_grid(Nx):
    Nnodes = Nx+1
    x = np.linspace(0, 1, Nnodes)
    y = np.linspace(0, 1, Nnodes)
    X, Y = np.meshgrid(x,y)
    nodes = np.zeros((Nnodes**2,2))
    nodes[:,0] = X.ravel()
    nodes[:,1] = Y.ravel()
    ID = np.zeros(len(nodes), dtype=np.int64)
    boundaries = dict()  # Will hold the boundary values
    n_eq = 0
    for nID in range(len(nodes)):
        if np.allclose(nodes[nID, 0], 0):
            pass
            # ID[nID] = -1
            # boundaries[nID] = 0  # Dirichlet BC
        else:
            ID[nID] = n_eq
            n_eq += 1
            if ( (np.allclose(nodes[nID, 1], 0)) or 
                 (np.allclose(nodes[nID, 0], 1)) or 
                 (np.allclose(nodes[nID, 1], 1)) or 
                 (np.allclose(nodes[nID, 0], 0)) ):
                boundaries[nID] = 0 # Neumann BC
    IEN = np.zeros((2*Nx**2, 3), dtype=np.int64)
    for i in range(Nx):
        for j in range(Nx):
            IEN[2*i+2*j*Nx  , :] = (i+j*Nnodes, 
                                    i+1+j*Nnodes, 
                                    i+(j+1)*Nnodes)
            IEN[2*i+1+2*j*Nx, :] = (i+1+j*Nnodes, 
                                    i+1+(j+1)*Nnodes, 
                                    i+(j+1)*Nnodes)
    return nodes, IEN, ID, boundaries

Nx = 7

nodes,IEN,ID,boundary_nodes = generate_2d_grid(Nx)

def find_node_by_coordinate(coordinate, nodes):
    coordinate = np.atleast_2d(coordinate)  # Ensure it's 2D for broadcasting
    distances = np.linalg.norm(nodes - coordinate, axis=1)  # Compute Euclidean distances
    index = np.where(distances < 1e-8)[0]  # Find indices with very small distance
    return index[0] if len(index) > 0 else -1

def distorted_grid():
    # Node coordinates
    nodes = np.array([
        [0.0, 0.0],  # Node 0
        [1.0, 0.0],  # Node 1
        [0.5, 0.8],  # Node 2
        [0.5, 0.2]   # Node 3
    ])

    # Element connectivity (IEN)
    IEN = np.array([
        [0, 1, 3],  # Element 0
        [3, 1, 2]   # Element 1
    ])

    # Boundary conditions
    boundary_nodes = {
        0: 0.0,  # Dirichlet BC at Node 0
        1: 0.0,  # Dirichlet BC at Node 1
        2: 1.0,  # Neumann BC at Node 2 (example value)
        3: 1.0   # Neumann BC at Node 3 (example value)
    }

    # ID Array for interior and boundary nodes
    ID = np.array([-1, -1, 0, 1])  # Assign -1 to boundary nodes
    return nodes,IEN,ID,boundary_nodes

#nodes,IEN,ID,boundary_nodes = distorted_grid()

# Make all boundary points Dirichlet
def S(X):
    X = np.atleast_2d(X)
    return 1.0
def exact(X):
    X = np.atleast_2d(X)
    x = X[:,0]
    y = X[:,1]
    return np.array(x*(1-x/2))    

def exact_point(X):
    x = X[0]
    y = X[1]
    return x*(1-x/2)

# def S(X):
#     X = np.atleast_2d(X)
#     return 1-X[:,0]
# def exact(X):
#     X = np.atleast_2d(X)
#     x = X[:,0]
#     y = X[:,1]
#     return np.array(x * (x**2 - 3 * x + 3) / 6)   

# def S(X):
#     X = np.atleast_2d(X)
#     return (1-X[:,0])**2
# def exact(X):
#     X = np.atleast_2d(X)
#     x = X[:,0]
#     y = X[:,1]
#     return np.array(x * (4 - 6 * x + 4 * x**2 - x**3) / 12)   

coordinates = np.array([[nodes[(Nx+1)*x,0],nodes[(Nx+1)*x,1]] for x in range(0,Nx+1)]) # Left side
#coordinates = np.array([[nodes[x,0],nodes[x,1]] for x in range(0,Nx+1)]) # right side
print(coordinates)
#coordinates = [0,2]
node_indices = [find_node_by_coordinate(coord, nodes) for coord in coordinates]
print(f"Node Indices: {node_indices}")  # Output: [3, 1]

n_eq = 0
for i in range(len(nodes[:, 1])):
    if i in node_indices:
        ID[i] = -1
    else:
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
        LM[a,e] = ID[IEN[e,a]]
# Global stiffness matrix and force vector
K = np.zeros((N_equations, N_equations))
F = np.zeros((N_equations,))
# Loop over elements
for e in range(N_elements):
    k_e = stiffness_matrix_for_element(nodes[IEN[e,:],:])
    f_e = force_vector_for_element(nodes[IEN[e,:],:], S)
    for a in range(3):
        A = LM[a, e]
        for b in range(3):
            B = LM[b, e]
            if (A >= 0) and (B >= 0):
                #print(f"Adding k_e[{a}, {b}] = {k_e[a, b]} to K[{A}, {B}]")
                K[A, B] += k_e[a, b]
        if (A >= 0):
            F[A] += f_e[a]
# Solve
Psi_interior = np.linalg.solve(K, F)
Psi_A = np.zeros(N_nodes)
for n in range(N_nodes):
    if ID[n] >= 0: # Otherwise Psi should be zero, and we've initialized that already.
        Psi_A[n] = Psi_interior[ID[n]]

for n in node_indices:
    Psi_A[n] = exact_point(nodes[n,:])  # Explicitly set the value        

plt.tripcolor(nodes[:,0], nodes[:,1], Psi_A, triangles=IEN)
plt.scatter(coordinates[:,0],coordinates[:,1],c='r')
plt.colorbar()
plt.show()
plt.tripcolor(nodes[:,0], nodes[:,1], exact(nodes), triangles=IEN)
plt.colorbar()
plt.show()
