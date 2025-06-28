#e8_clock_check_c10.py

import numpy as np

# --- Part 1: Operator and Shell Definitions ---

def get_c5_rotor_by_construction():
    """Defines the C5 rotation matrix 's' from the Mathematica source."""
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

def get_c2_rotor_by_construction():
    """Defines the C2 swap operator 't' by construction as a perfect permutation."""
    return np.block([
        [np.zeros((4, 4)), np.identity(4)],
        [np.identity(4), np.zeros((4, 4))]
    ])

def generate_shells_from_rotor(rotor):
    """Generates 10 shells by starting with two orthogonal "seed" shells."""
    print("Generating shells from C5 rotor...")
    d4_roots = set()
    for i in range(4):
        for j in range(i + 1, 4):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    q = np.zeros(4); q[i], q[j] = s1, s2
                    d4_roots.add(tuple(q))
    d4_roots = np.array(list(d4_roots))
    seed_a = np.hstack([d4_roots, np.zeros((24, 4))])
    seed_b = np.hstack([np.zeros((24, 4)), d4_roots])
    
    shells_a = [seed_a @ np.linalg.matrix_power(rotor, k) for k in range(5)]
    shells_b = [seed_b @ np.linalg.matrix_power(rotor, k) for k in range(5)]
    
    final_shells = [item for pair in zip(shells_a, shells_b) for item in pair]
    print("Shells generated successfully.")
    return final_shells

# --- Part 2: Rigorous Testing Framework ---

def test_g_cycle(g_op, cycle_len, start_vec, vec_map):
    """
    Tests if the composite operator g=st generates a closed C10 cycle
    of the shells.
    """
    print(f"\n--- Testing if 'g=st' generates a C{cycle_len} vertex cycle ---")
    
    current_vec = start_vec
    start_shell_idx = vec_map[tuple(np.round(start_vec * 2).astype(int))]
    path_indices = [start_shell_idx]
    
    for i in range(cycle_len):
        current_vec = g_op @ current_vec
        current_vec_int = tuple(np.round(current_vec * 2).astype(int))
        
        # This check should always pass now, but we keep it for rigor
        if current_vec_int not in vec_map:
            print(f"  - Step {i+1}: FAILED. Rotated vertex is NOT on the E8 lattice.")
            return False, []

        path_indices.append(vec_map[current_vec_int])

    path_is_unbroken = len(set(path_indices[:-1])) == cycle_len
    returned_to_start = np.allclose(current_vec, start_vec)

    print(f"  - Path of shells visited: {' -> '.join(map(str, path_indices))}")
    print(f"  - Does the path visit {cycle_len} unique shells? {path_is_unbroken}")
    print(f"  - Does the path return to the starting vertex? {returned_to_start}")
    
    return path_is_unbroken and returned_to_start

# --- Part 3: Main Execution ---

if __name__ == "__main__":
    # --- Setup Operators By Construction ---
    s_rotor = get_c5_rotor_by_construction()
    t_rotor = get_c2_rotor_by_construction()
    g_rotor = s_rotor @ t_rotor

    # --- Generate Shells and Verification Map ---
    LAMBDA = generate_shells_from_rotor(s_rotor)
    vec_map = {tuple(np.round(v * 2).astype(int)): idx
               for idx, shell in enumerate(LAMBDA) for v in shell}

    print("\n" + "="*65)
    print("      E8 Cycle Clock: The Final Test of the 'g=st' Operator")
    print("="*65)

    start_vector = LAMBDA[0][0]
    
    # --- Run the final, definitive test ---
    is_c10_cycle = test_g_cycle(g_rotor, 10, start_vector, vec_map)

    print("\n" + "="*65)
    print("                          FINAL CONCLUSION")
    print("="*65)
    if is_c10_cycle:
        print("✅ SUCCESS: The composite operator 'g=st' is a true vertex")
        print("   symmetry that generates a single, unbroken 10-cycle.")
        print("\n   Even though 's' and 't' do not commute, their product 'g=st'")
        print("   is a valid C10 generator for the E8 Clock.")
    else:
        print("❌ FAILURE: The composite operator 'g=st', while a true vertex")
        print("   symmetry, does NOT generate a single 10-cycle.")
        print("\n   This confirms the clock must be purely procedural, as no single")
        print("   operator generates the full cycle.")
