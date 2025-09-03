import numpy as np
from scipy.linalg import null_space

# ==============================================================================
# E8 Clock Model: Computational Analysis
#
# This script performs a computational analysis of a clock model on the E8
# root system based on a C5 rotational symmetry ('s') and a C2 subspace
# swap ('t'). It calculates the properties and stabilizer dimensions of
# these operators.
#
# Author: Marcelo Amaral
# Computational Implementation: Google Gemini
# Date: June 30, 2025
# ==============================================================================

# --- Part 1: Operator and Shell Definitions ---

def get_c5_rotor():
    """Defines the C5 rotation matrix 's' by construction."""
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

def get_c2_rotor():
    """Defines the C2 swap operator 't' by construction."""
    return np.block([
        [np.zeros((4, 4)), np.identity(4)],
        [np.identity(4), np.zeros((4, 4))]
    ])

def generate_e8_shells(rotor):
    """Generates 10 shells from two orthogonal seed shells."""
    d4_roots_set = set()
    for i in range(4):
        for j in range(i + 1, 4):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    q = np.zeros(4); q[i], q[j] = s1, s2
                    d4_roots_set.add(tuple(q))
    d4_roots = np.array(list(d4_roots_set))

    seed_a = np.hstack([d4_roots, np.zeros((24, 4))])
    seed_b = np.hstack([np.zeros((24, 4)), d4_roots])
    
    shells_a = [seed_a @ np.linalg.matrix_power(rotor, k) for k in range(5)]
    shells_b = [seed_b @ np.linalg.matrix_power(rotor, k) for k in range(5)]
    
    return [item for pair in zip(shells_a, shells_b) for item in pair]

# --- Part 2: Analysis Functions ---

def get_so8_lie_algebra_basis():
    """Returns a standard basis of 28 generators for so(8)."""
    dim = 8
    basis = []
    for i in range(dim):
        for j in range(i + 1, dim):
            G = np.zeros((dim, dim)); G[i, j] = 1; G[j, i] = -1
            basis.append(G)
    return basis

def calculate_stabilizer_dimension(operators, algebra_basis):
    """Calculates the dimension of the stabilizer Lie algebra."""
    if not isinstance(operators, list):
        operators = [operators]
    list_of_ad_matrices = []
    for R in operators:
        ad_R_cols = [(R @ Gk - Gk @ R).flatten() for Gk in algebra_basis]
        list_of_ad_matrices.append(np.array(ad_R_cols).T)
    M = np.vstack(list_of_ad_matrices)
    rank_M = np.linalg.matrix_rank(M, tol=1e-9)
    return len(algebra_basis) - rank_M

def find_true_cycle_path(op, start_vec, vec_map):
    """Discovers the permutation path of an operator on the shells."""
    path = [vec_map.get(tuple(np.round(start_vec, 5)))]
    current_vec = np.copy(start_vec)
    for _ in range(20): # Loop enough times to find a cycle
        current_vec = current_vec @ op
        path.append(vec_map.get(tuple(np.round(current_vec, 5))))
        if np.allclose(current_vec, start_vec):
            return " -> ".join(map(str, path))
    return "No simple cycle found."

# --- Part 3: Main Execution and Report ---
if __name__ == "__main__":
    print("Performing computational analysis of the E8 C5/C2 model...")
    
    # 1. Define operators and generate shells
    s_rotor = get_c5_rotor()
    t_rotor = get_c2_rotor()
    LAMBDA = generate_e8_shells(s_rotor)
    
    # 2. Setup for analysis
    vec_map = {tuple(np.round(v, 5)): idx for idx, s in enumerate(LAMBDA) for v in s}
    so8_basis = get_so8_lie_algebra_basis()

    # 3. Perform calculations
    are_commuting = np.allclose(s_rotor @ t_rotor, t_rotor @ s_rotor)
    
    s_cycle_path_even = find_true_cycle_path(s_rotor, LAMBDA[0][0], vec_map)
    s_cycle_path_odd = find_true_cycle_path(s_rotor, LAMBDA[1][0], vec_map)
    t_cycle_path = find_true_cycle_path(t_rotor, LAMBDA[0][0], vec_map)
    
    stab_dim_s = calculate_stabilizer_dimension(s_rotor, so8_basis)
    stab_dim_t = calculate_stabilizer_dimension(t_rotor, so8_basis)
    stab_dim_intersection = calculate_stabilizer_dimension([s_rotor, t_rotor], so8_basis)

    # 4. Print the results report
    print("\n" + "="*70)
    print("      COMPUTATIONAL ANALYSIS REPORT")
    print("="*70)
    
    print("\n## Operator Properties")
    print(f"  - Operators 's' and 't' commute: {are_commuting}")
    
    print("\n## C5 Operator ('s') Analysis")
    print(f"  - Action on even shells: {s_cycle_path_even}")
    print(f"  - Action on odd shells:  {s_cycle_path_odd}")
    print(f"  - Stabilizer dimension dim(Stab(s)): {stab_dim_s}")

    print("\n## C2 Operator ('t') Analysis")
    print(f"  - Action on shell 0: {t_cycle_path}")
    print(f"  - Stabilizer dimension dim(Stab(t)): {stab_dim_t}")

    print("\n## Combined System Analysis")
    print(f"  - Stabilizer dimension dim(Stab(s) ∩ Stab(t)): {stab_dim_intersection}")
    
    print("\n" + "="*70)