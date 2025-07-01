import numpy as np

# --- Part 1: Operator and Shell Definitions ---

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

def get_sigma_rotor():
    """Defines the 'σ' C4 operator (The "Slow Clicker")."""
    I4 = np.identity(4)
    Z4 = np.zeros((4, 4))
    # This corresponds to σ = ArrayFlatten[{{0, Id}, {-Id, 0}}]
    return np.block([
        [Z4, I4],
        [-I4, Z4]
    ])

def generate_shells(s_op, sigma_op):
    """Generates 10 shells using the 'Fast Pointer, Slow Clicker' model."""
    d4_roots_set = set()
    for i in range(4):
        for j in range(i + 1, 4):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    q = np.zeros(4); q[i], q[j] = s1, s2
                    d4_roots_set.add(tuple(q))
    d4_roots = np.array(list(d4_roots_set))
    seed_a = np.hstack([d4_roots, np.zeros((24, 4))])
    
    shells_A = [seed_a @ np.linalg.matrix_power(s_op, k) for k in range(5)]
    shells_B = [(seed_a @ sigma_op) @ np.linalg.matrix_power(s_op, k) for k in range(5)]
    
    # Interleave them to match our {0,2,4..} and {1,3,5..} cycle convention
    return [item for pair in zip(shells_A, shells_B) for item in pair]

def are_clouds_equal(cloud1, cloud2):
    """Checks if two point clouds are identical, regardless of order."""
    if cloud1.shape != cloud2.shape: return False
    sorted1 = cloud1[np.lexsort(cloud1.T)]
    sorted2 = cloud2[np.lexsort(cloud2.T)]
    return np.allclose(sorted1, sorted2)

# --- Part 2: Analysis Function ---

def trace_fast_slow_clock(s_op, sigma_op, shells):
    """Traces the full path of the 'Fast Pointer, Slow Clicker' clock."""
    print("\n## Testing the 'Fast Pointer, Slow Clicker' Procedural Path")
    
    # Create a lookup map for shell identification
    vec_map = {tuple(np.round(v, 5)): idx for idx, s in enumerate(shells) for v in s}
    
    path = [0]
    current_shell_idx = 0
    
    # Trace the first 5-cycle
    print("  - Tracing first 5-cycle with 's' (fast pointer)...")
    for _ in range(5):
        current_shell = shells[current_shell_idx]
        transformed_shell = current_shell @ s_op
        for next_idx in range(10):
            if are_clouds_equal(transformed_shell, shells[next_idx]):
                current_shell_idx = next_idx
                path.append(current_shell_idx)
                break
    print(f"    - Path after 5 's' steps: {' -> '.join(map(str, path))}")
    
    # Perform the "Slow Click"
    print("  - Performing 'slow click' with 'σ' to jump cycles...")
    current_shell = shells[current_shell_idx]
    transformed_shell = current_shell @ sigma_op
    for next_idx in range(10):
        if are_clouds_equal(transformed_shell, shells[next_idx]):
            current_shell_idx = next_idx
            path.append(current_shell_idx)
            break
    print(f"    - Path after 'σ' click: {' -> '.join(map(str, path))}")

    # Trace the second 5-cycle
    print("  - Tracing second 5-cycle with 's'...")
    for _ in range(5):
        current_shell = shells[current_shell_idx]
        transformed_shell = current_shell @ s_op
        for next_idx in range(10):
            if are_clouds_equal(transformed_shell, shells[next_idx]):
                current_shell_idx = next_idx
                path.append(current_shell_idx)
                break
    
    print("\n  - **Final Path:**")
    print(f"    `{' -> '.join(map(str, path))}`")
    print(f"  - **Unique Shells Visited:** {len(set(path))}")
    
    if len(set(path)) == 10:
        print("  - ✅ SUCCESS: The procedure visits all 10 unique shells.")
    else:
        print("  - ❌ FAILED: The procedure does not visit all shells.")

# --- Part 3: Main Execution ---
if __name__ == "__main__":
    s_rotor = get_c5_rotor()
    sigma_rotor = get_sigma_rotor()
    
    LAMBDA = generate_shells(s_rotor, sigma_rotor)

    print("\n" + "="*70)
    print("      VERIFICATION OF THE PROCEDURAL CLOCK")
    print("="*70)

    trace_fast_slow_clock(s_rotor, sigma_rotor, LAMBDA)

    print("\n======================================================================")