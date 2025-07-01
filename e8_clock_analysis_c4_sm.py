import numpy as np
from scipy.linalg import null_space

# --- Part 1: Operator Definitions ---

def get_c5_rotor():
    """Defines the C5 rotation matrix 's' (The "Fast Pointer")."""
    s_op = np.array([
        [-0.5,  0.5,  0.0,  0.0,  0.0,  0.0,  0.5,  0.5],
        [-0.5, -0.5,  0.0,  0.0,  0.0,  0.0,  0.5, -0.5],
        [ 0.0,  0.0, -0.5,  0.5, -0.5, -0.5,  0.0,  0.0],
        [ 0.0,  0.0, -0.5, -0.5, -0.5,  0.5,  0.0,  0.0],
        [-0.5,  0.5,  0.0,  0.0,  0.0,  0.0, -0.5, -0.5],
        [-0.5, -0.5,  0.0,  0.0,  0.0,  0.0, -0.5,  0.5],
        [ 0.0,  0.0, -0.5,  0.5,  0.5,  0.5,  0.0,  0.0],
        [ 0.0,  0.0, -0.5, -0.5,  0.5, -0.5,  0.0,  0.0]
    ])
    return s_op.T

def get_c4_sigma_rotor():
    """Defines the 'σ' C4 operator (The "Slow Clicker")."""
    I4 = np.identity(4)
    Z4 = np.zeros((4, 4))
    return np.block([[Z4, I4], [-I4, Z4]])

# --- Part 2: Analysis Function ---

def get_so8_lie_algebra_basis():
    """Returns a standard basis of 28 generators for so(8)."""
    dim = 8
    basis = []
    for i in range(dim):
        for j in range(i + 1, dim):
            G = np.zeros((dim, dim)); G[i, j] = 1; G[j, i] = -1
            basis.append(G)
    return basis

def get_stabilizer_basis(operator, algebra_basis):
    """
    Finds the explicit matrix basis for the stabilizer Lie algebra.
    """
    # Create the constraint matrix M where Mv=0
    ad_R_cols = [(operator @ Gk - Gk @ operator).flatten() for Gk in algebra_basis]
    M = np.array(ad_R_cols).T
    
    # Find the null space of M. The columns of null_space_matrix are the
    # coefficients of the basis vectors for the stabilizer.
    null_space_matrix = null_space(M)
    
    # Reconstruct the 8x8 basis matrices from the coefficients
    stabilizer_basis_matrices = []
    for i in range(null_space_matrix.shape[1]):
        # Get the i-th basis vector for the null space (as a column)
        coeffs = null_space_matrix[:, i]
        # Reconstruct the 8x8 matrix by summing the so(8) basis generators
        # weighted by the coefficients.
        basis_matrix = sum(c * Gk for c, Gk in zip(coeffs, algebra_basis))
        stabilizer_basis_matrices.append(basis_matrix)
        
    return stabilizer_basis_matrices

# --- Part 3: Main Execution and Report ---

if __name__ == "__main__":
    sigma_rotor = get_c4_sigma_rotor()
    so8_basis = get_so8_lie_algebra_basis()

    print("\n" + "="*70)
    print("      EXTRACTING THE BASIS FOR THE STABILIZER OF σ")
    print("="*70)

    # Calculate the basis for the stabilizer of sigma
    stab_sigma_basis = get_stabilizer_basis(sigma_rotor, so8_basis)
    
    print(f"\nFound a basis for Stab(σ) with **{len(stab_sigma_basis)}** generators.")
    print("These 16 matrices form the Lie algebra to be analyzed.")
    
    # Optionally print the basis matrices
    # for i, B in enumerate(stab_sigma_basis):
    #     print(f"\n--- Basis Generator {i+1} ---")
    #     print(np.round(B, 2))

    print("\n" + "="*70)