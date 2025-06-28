#e8_clock_check.py

import numpy as np
from scipy.linalg import orthogonal_procrustes

# --- Part 1: Operator and Shell Definitions ---

def get_mathematica_c5_rotor():
    """Defines the C5 rotation matrix 's'."""
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

def generate_shells_from_rotor(rotor):
    """
    Generates 10 shells by starting with two orthogonal "seed" shells
    and rotating them with the provided C5 operator 's'.
    """
    print("Generating shells from C5 rotor...")
    d4_roots = set()
    for i in range(4):
        for j in range(i + 1, 4):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    q = np.zeros(4); q[i], q[j] = s1, s2
                    d4_roots.add(tuple(q))
    d4_roots = np.array(list(d4_roots))
    # Seed A lives in the first 4 coordinates
    seed_a = np.hstack([d4_roots, np.zeros((24, 4))])
    # Seed B lives in the last 4 coordinates
    seed_b = np.hstack([np.zeros((24, 4)), d4_roots])
    
    shells_a = [seed_a @ np.linalg.matrix_power(rotor, k) for k in range(5)]
    shells_b = [seed_b @ np.linalg.matrix_power(rotor, k) for k in range(5)]
    
    final_shells = [item for pair in zip(shells_a, shells_b) for item in pair]
    print("Shells generated successfully.")
    return final_shells

# --- Part 2: Main Execution ---

if __name__ == "__main__":
    # --- Setup Operators By Construction ---
    s_rotor = get_mathematica_c5_rotor()
    
    # Define 't' by construction as a perfect swap of the two 4D subspaces.
    t_rotor = np.block([
        [np.zeros((4, 4)), np.identity(4)],
        [np.identity(4), np.zeros((4, 4))]
    ])

    print("\n--- s_rotor @ t_rotor ---")
    print(s_rotor @ t_rotor)

    print("\n--- t_rotor @ s_rotor ---")
    print(t_rotor @ s_rotor)

    print("\n--- Difference (s @ t - t @ s) ---")
    print(s_rotor @ t_rotor - t_rotor @ s_rotor)


    print("\n--- Testing Foundational Commutativity ---")
    commutes = np.allclose(s_rotor @ t_rotor, t_rotor @ s_rotor)
    print(f"Do s_rotor and t_rotor (defined by construction) commute? {commutes}")
    if not commutes:
        print("CRITICAL FAILURE: The C5 and C2 operators do not commute.")
        print("The system CANNOT form a C5 x C2 group.")


    g_rotor = s_rotor @ t_rotor

    # --- Generate Shells and Verification Map ---
    LAMBDA = generate_shells_from_rotor(s_rotor)
    vec_map = {tuple(np.round(v * 2).astype(int)): idx
               for idx, shell in enumerate(LAMBDA) for v in shell}

    print("\n" + "="*65)
    print("      Explicit Coordinate Check for g=st Vertex Symmetry")
    print("="*65)

    # --- Perform the explicit check ---
    start_vector = LAMBDA[0][0]
    rotated_vector = g_rotor @ start_vector
    
    # For printing, set small values to zero for clarity
    np.set_printoptions(precision=4, suppress=True)

    print("\n1. Starting with a known E8 vertex from Shell 0:")
    print(f"   Start Vector: {start_vector}")

    print("\n2. Applying the composite rotation g = st:")
    print(f"   Rotated Vector: {rotated_vector}")
    
    # Check if the result is a valid E8 vertex
    rotated_vector_int = tuple(np.round(rotated_vector * 2).astype(int))
    is_in_lattice = rotated_vector_int in vec_map

    print("\n3. Verification:")
    print(f"   Is the rotated vector found anywhere in the original 240 E8 vertices?")
    print(f"   RESULT: {is_in_lattice}")

    print("\n" + "="*65)
    print("                          FINAL CONCLUSION")
    print("="*65)
    if not is_in_lattice:
        print("✅ The coordinates of the rotated vector are not half-integers.")
        print("   This confirms it is NOT a point on the E8 root lattice.")
        print("   Therefore, 'g=st' is not a vertex symmetry, and the clock")
        print("   must be procedural.")
    else:
        # This case should not be reached
        print("❌ An unexpected result occurred. The rotated vector is in the lattice.")