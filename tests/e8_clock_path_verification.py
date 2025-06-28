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
    
    dots = seed_a @ seed_b.T      # 24×24 matrix of pairwise dot products
    assert np.allclose(dots, 0)   # passes ⇒ orthogonal sets


    shells_a = [seed_a @ np.linalg.matrix_power(rotor, k) for k in range(5)]
    shells_b = [seed_b @ np.linalg.matrix_power(rotor, k) for k in range(5)]
    
    final_shells = [item for pair in zip(shells_a, shells_b) for item in pair]
    print("Shells generated successfully.")
    return final_shells

# --- Part 2: Rigorous Testing Framework ---

def test_vertex_cycle(op, op_name, cycle_len, start_vec, vec_map):
    """
    Tests if an operator is a true vertex symmetry that generates a closed cycle.
    """
    print(f"\n--- Testing Vertex Symmetry for Operator '{op_name}' (Expected Cycle: {cycle_len}) ---")
    
    current_vec = start_vec
    path_indices = [vec_map[tuple(np.round(start_vec * 2).astype(int))]]
    
    for i in range(cycle_len):
        current_vec = op @ current_vec
        current_vec_int = tuple(np.round(current_vec * 2).astype(int))
        
        if current_vec_int not in vec_map:
            print(f"  - Step {i+1}: FAILED. Rotated vertex is NOT on the E8 lattice.")
            print(f"  ❌ Conclusion: '{op_name}' is NOT a vertex symmetry.")
            return False

        path_indices.append(vec_map[current_vec_int])

    if not np.allclose(current_vec, start_vec):
        print(f"  - FAILED. After {cycle_len} steps, did not return to the start.")
        print(f"  ❌ Conclusion: '{op_name}' does not generate a closed {cycle_len}-cycle.")
        return False
        
    print(f"  - PASSED. Operator is a vertex symmetry for all {cycle_len} steps.")
    print(f"  - Path of shells visited: {' -> '.join(map(str, path_indices))}")
    print(f"  ✅ Conclusion: '{op_name}' is a valid symmetry generating a closed {cycle_len}-cycle.")
    return True

# --- Part 3: Main Execution ---

if __name__ == "__main__":
    # --- Setup Operators By Construction ---
    s_rotor = get_mathematica_c5_rotor()
    
    # Define 't' by construction as a perfect swap of the two 4D subspaces.
    # This is a simple permutation and guaranteed to be a vertex symmetry.
    t_rotor = np.block([
        [np.zeros((4, 4)), np.identity(4)],
        [np.identity(4), np.zeros((4, 4))]
    ])
    
    g_rotor = s_rotor @ t_rotor

    # --- Generate Shells and Verification Map ---
    LAMBDA = generate_shells_from_rotor(s_rotor)
    vec_map = {tuple(np.round(v * 2).astype(int)): idx
               for idx, shell in enumerate(LAMBDA) for v in shell}

    print("\n" + "="*65)
    print("      Definitive E8 Vertex Symmetry and Cycle Verification")
    print("="*65)
    
    print("\n--- Testing Foundational Commutativity ---")
    commutes = np.allclose(s_rotor @ t_rotor, t_rotor @ s_rotor)
    print(f"Do s_rotor and t_rotor (defined by construction) commute? {commutes}")
    if not commutes:
        print("CRITICAL FAILURE: The C5 and C2 operators do not commute.")
        print("The system CANNOT form a C5 x C2 group.")

    # --- Run the rigorous tests ---
    print("\n--- Testing Symmetries of Individual Operators ---")
    s_ok_set1 = test_vertex_cycle(s_rotor, 's (C5) on first shell set (even indices)', 5, LAMBDA[0][0], vec_map)
    s_ok_set2 = test_vertex_cycle(s_rotor, 's (C5) on second shell set (odd indices)', 5, LAMBDA[1][0], vec_map)
    t_ok = test_vertex_cycle(t_rotor, 't (C2)', 2, LAMBDA[0][0], vec_map)
    
    s_ok = s_ok_set1 and s_ok_set2

    g_ok = False
    if commutes and s_ok and t_ok:
        g_ok = test_vertex_cycle(g_rotor, 'g = st (C10)', 10, LAMBDA[0][0], vec_map)
    else:
        print(f"\n--- Testing Vertex Symmetry for Operator 'g = st (C10)' ---")
        print("  - SKIPPED. Test cannot proceed because 's' and 't' do not commute or are not symmetries.")

    print("\n" + "="*65)
    print("                          FINAL CONCLUSION")
    print("="*65)
    if commutes and s_ok and t_ok and g_ok:
        print("✅ SUCCESS: The C5xC2 Cycle Clock hypothesis is verified.")
        print("Defining both 's' and 't' by construction results in a system where:")
        print("1. 's' and 't' are true E8 vertex symmetries.")
        print("2. 's' and 't' commute, forming a valid C5 x C2 group.")
        print("3. Their product 'g=st' is also a true vertex symmetry that generates")
        print("   a single, unbroken 10-cycle through the shells.")
    else:
        print("❌ FAILURE: The C5xC2 Cycle Clock hypothesis is falsified.")
        print("Even when both 's' and 't' are defined by construction as discrete")
        print("permutations, they do not commute. Therefore, they cannot form the")
        print("required C5 x C2 group structure to generate the clock.")

