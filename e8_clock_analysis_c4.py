import numpy as np

# ==============================================================================
# E8 "Fast Pointer, Slow Clicker" Clock: Stabilizer Analysis
#
# This script analyzes the stabilizer subgroups of the working E8 clock model,
# which is based on a C5 rotational symmetry ('s') and a C4 bridging
# symmetry ('σ').
#
# Author: Marcelo Amaral, Klee Irwin
# Computational Implementation: Google Gemini
# Date: June 30, 2025
# ==============================================================================

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
    # This corresponds to σ = ArrayFlatten[{{0, Id}, {-Id, 0}}]
    return np.block([
        [Z4, I4],
        [-I4, Z4]
    ])

# --- Part 2: Analysis Function ---

def get_so8_lie_algebra_basis():
    """Returns a standard basis of 28 generators for so(8)."""
    dim = 8
    basis = []
    for i in range(dim):
        for j in range(i + 1, dim):
            G = np.zeros((dim, dim))
            G[i, j] = 1
            G[j, i] = -1
            basis.append(G)
    return basis

def calculate_stabilizer_dimension(operators, algebra_basis):
    """Calculates the dimension of the stabilizer Lie algebra (centralizer)."""
    if not isinstance(operators, list):
        operators = [operators]
    
    # Create the constraint matrix M where Mv=0
    list_of_ad_matrices = []
    for R in operators:
        # Each row of ad_R defines the commutator [R, G_k]
        ad_R_cols = [(R @ Gk - Gk @ R).flatten() for Gk in algebra_basis]
        list_of_ad_matrices.append(np.array(ad_R_cols).T)
    
    M = np.vstack(list_of_ad_matrices)
    
    # The dimension of the stabilizer is the dimension of the null space of M.
    # This is n - rank(M), where n is the number of generators.
    rank_M = np.linalg.matrix_rank(M, tol=1e-9)
    return len(algebra_basis) - rank_M

# --- Part 3: Main Execution and Report ---

if __name__ == "__main__":
    # 1. Define the clock's component operators
    s_rotor = get_c5_rotor()
    sigma_rotor = get_c4_sigma_rotor()
    so8_basis = get_so8_lie_algebra_basis()

    print("\n" + "="*70)
    print("      STABILIZER ANALYSIS OF THE 'FAST POINTER, SLOW CLICKER' CLOCK")
    print("="*70)

    # 2. Calculate the dimension of the stabilizer for each component
    stab_dim_s = calculate_stabilizer_dimension(s_rotor, so8_basis)
    stab_dim_sigma = calculate_stabilizer_dimension(sigma_rotor, so8_basis)
    stab_dim_intersection = calculate_stabilizer_dimension([s_rotor, sigma_rotor], so8_basis)

    # 3. Print the results
    print(f"\n## 1. Stabilizer of the C5 'Fast Pointer' (s)")
    print(f"   - The dimension of Stab(s) is: **{stab_dim_s}**")

    print(f"\n## 2. Stabilizer of the C4 'Slow Clicker' (σ)")
    print(f"   - The dimension of Stab(σ) is: **{stab_dim_sigma}**")

    print(f"\n## 3. Stabilizer of the Full Clock (s and σ)")
    print(f"   - The dimension of Stab(s) ∩ Stab(σ) is: **{stab_dim_intersection}**")
    
    print("\n" + "="*70)
    print("                          FINAL CONCLUSION")
    print("="*70)
    print("The key result is the dimension of the stabilizer of the 'clicker' (σ).")
    print("This represents the symmetry that is preserved when jumping between the")
    print("two 5-cycles and is the primary candidate for the Standard Model group.")
    print(f"\nThis dimension is **{stab_dim_sigma}**. Further analysis is required to determine if its")
    print("Lie algebra structure matches that of SU(3)xSU(2)xU(1).")
    print("="*70)