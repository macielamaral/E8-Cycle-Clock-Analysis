import numpy as np
from scipy.linalg import orthogonal_procrustes

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

def get_base_c2_rotor():
    """Defines the base C2 swap operator 't_0'."""
    return np.block([
        [np.zeros((4, 4)), np.identity(4)],
        [np.identity(4), np.zeros((4, 4))]
    ])

def generate_e8_shell_partition(c5_rotor):
    """Generates the 10 D4-shells of E8."""
    print("Generating E8 shell partition...")
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
    
    shells_a = [seed_a @ np.linalg.matrix_power(c5_rotor, k) for k in range(5)]
    shells_b = [seed_b @ np.linalg.matrix_power(c5_rotor, k) for k in range(5)]

    final_shells = [item for pair in zip(shells_a, shells_b) for item in pair]
    print("Shell partition generated successfully.")
    return final_shells

# --- Part 2: "Multiple C2s" Clock Logic ---

def generate_c2_family(s_op, t0_op):
    """Generates the family of t_k operators via conjugation."""
    print("\nGenerating the family of 5 C2 operators via conjugation...")
    t_family = []
    for k in range(5):
        s_k = np.linalg.matrix_power(s_op, k)
        s_k_inv = np.linalg.matrix_power(s_op, -k)
        t_k = s_k @ t0_op @ s_k_inv
        t_family.append(t_k)
    print("C2 family generated.")
    return t_family

def verify_local_swaps(t_family, shells, vec_map):
    """
    Verifies that each t_k correctly swaps its designated shell pair in
    both directions (e.g., 0->1 and 1->0).
    """
    print("\n## Verifying Local Swaps for each C2 Operator (Both Directions)")
    all_swaps_correct = True
    
    for k in range(5):
        # Test the even -> odd direction
        even_start_idx = 2 * k
        odd_expected_idx = even_start_idx + 1
        
        start_vec_even = shells[even_start_idx][0]
        end_vec_even = start_vec_even @ t_family[k]
        end_idx_even = vec_map.get(tuple(np.round(end_vec_even, 5)), -1)

        print(f"  - Testing t_{k}: Shell `{even_start_idx}` -> Shell `{end_idx_even}`", end="")
        if end_idx_even == odd_expected_idx:
            print(" (Correct)")
        else:
            print(" (ERROR!)")
            all_swaps_correct = False

        # Test the odd -> even direction
        odd_start_idx = 2 * k + 1
        even_expected_idx = odd_start_idx - 1

        start_vec_odd = shells[odd_start_idx][0]
        end_vec_odd = start_vec_odd @ t_family[k]
        end_idx_odd = vec_map.get(tuple(np.round(end_vec_odd, 5)), -1)

        print(f"  - Testing t_{k}: Shell `{odd_start_idx}` -> Shell `{end_idx_odd}`", end="")
        if end_idx_odd == even_expected_idx:
            print(" (Correct)")
        else:
            print(" (ERROR!)")
            all_swaps_correct = False
        
        if k < 4: print("  " + "-"*20) # Add a separator

    if all_swaps_correct:
        print("\n  - ✅ **Conclusion:** All local C2 operators correctly swap their respective shell pairs.")
    else:
        print("\n  - ❌ **Conclusion:** There is an error in the local swap logic.")

def trace_sophisticated_clock(t_family, s_op, start_vec, vec_map):
    """Traces the full 10-cycle using the appropriate t_k at each step."""
    print("\n## Tracing the Sophisticated Procedural Clock")
    path_indices = [vec_map[tuple(np.round(start_vec, 5))]]
    path_ops = ["Start"]
    current_vec = np.copy(start_vec)

    for i in range(10):
        if i % 2 == 0:  # Swap step
            k = i // 2
            op = t_family[k]
            path_ops.append(f"t_{k}")
        else:  # Rotation step
            op = s_op
            path_ops.append("s")
            
        current_vec = current_vec @ op
        path_indices.append(vec_map[tuple(np.round(current_vec, 5))])

    path_str = " -> ".join([f"{path_indices[i]} ({path_ops[i+1]})" for i in range(10)])
    print(f"  - **Path:** `{path_indices[0]} -> {path_str} -> {path_indices[10]}`")
    unique_visited = len(set(path_indices[:-1]))
    print(f"  - **Unique Shells Visited:** {unique_visited}")
    if unique_visited == 10 and path_indices[10] == path_indices[0]:
        print("  - ✅ **Conclusion:** The clock is **transitive** and completes a perfect 10-cycle.")
    else:
        print("  - ❌ **Conclusion:** The clock failed to complete a transitive 10-cycle.")

# --- Part 3: Main Execution ---
if __name__ == "__main__":
    s_rotor = get_c5_rotor()
    t0_rotor = get_base_c2_rotor()
    
    LAMBDA = generate_e8_shell_partition(s_rotor)
    vec_map = {tuple(np.round(v, 5)): idx for idx, s in enumerate(LAMBDA) for v in s}

    print("\n" + "="*70)
    print("      E8 CLOCK: 'MULTIPLE C2s' MODEL VERIFICATION")
    print("="*70)

    # 1. Generate the family of C2 operators
    t_family = generate_c2_family(s_rotor, t0_rotor)

    # 2. Verify that each C2 operator performs the correct local swap
    verify_local_swaps(t_family, LAMBDA, vec_map)
    print("-" * 70)

    # 3. Trace the full clock using the sophisticated procedure
    trace_sophisticated_clock(t_family, s_rotor, LAMBDA[0][0], vec_map)
    print("="*70)


    # Assume s_rotor, t_rotor, LAMBDA, and vec_map are already defined
# from the previous script.

# --- 1. Find the operator that swaps Shell 6 and 7 ---
print("## 1. Finding the specific C2 operator to swap Shells 6 and 7")

# Isolate the two shells
L6 = LAMBDA[6]
L7 = LAMBDA[7]

# Use the Procrustes solution to find the optimal rotation
# that maps the vertices of L6 to L7.
R, scale = orthogonal_procrustes(L6, L7)
t_fix = R
print(" - Found a candidate 't_fix' operator.")

# --- Verification Step ---
# Apply the new operator to Shell 6
transformed_L6 = L6 @ t_fix

# To check if the sets are equal regardless of vector order,
# we can sort the rows of each matrix and compare.
# A canonical way to sort rows is lexicographically.
sorted_transformed_L6 = transformed_L6[np.lexsort(transformed_L6.T)]
sorted_L7 = L7[np.lexsort(L7.T)]

# Check if the sorted matrices are identical
if np.allclose(sorted_transformed_L6, sorted_L7):
    print(" - ✅ SUCCESS: The new operator 't_fix' correctly maps Shell 6 to Shell 7.")
else:
    print(" - ❌ FAILURE: The new operator did not work.")


# --- 2. Test for Perpendicularity ---
print("\n## 2. Testing if Shell 6 and Shell 7 are perpendicular")

# Calculate the 24x24 matrix of all dot products
dot_product_matrix = L6 @ L7.T

# If they are perpendicular, this matrix should be all zeros.
is_perpendicular = np.allclose(dot_product_matrix, 0)

print(f" - Are Shells 6 and 7 perpendicular? **{is_perpendicular}**")