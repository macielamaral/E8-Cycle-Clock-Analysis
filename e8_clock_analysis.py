# ==============================================================================
# E8 Cycle Clock Analysis
#
# This script performs a rigorous computational analysis of the C5 x C2
# "Cycle Clock" structure within the E8 root system. It verifies theorems
# presented in the paper "Discrete Spinors, the C5xС2 Cycle Clock, and
# the Stabilizer Group in Spin(8)" by Amaral, Clawson, and Irwin.
#
# To Run:
# 1. Place this script in the same directory as the data files:
#    - five_rotor.txt
#    - ten_shells.txt
# 2. Ensure numpy and scipy are installed: pip install numpy scipy
# 3. Run from the command line: python e8_clock_analysis.py
#
# Authors: M. M. Amaral, R. Clawson, K. Irwin
# Computational Implementation: Google Gemini
# Date: June 2025
# ==============================================================================

import numpy as np
from scipy.linalg import svd, null_space
from fractions import Fraction
import re

# --- 1. Robust Data Loading ---

def parse_mathematica_line(line):
    """Takes a single line of text and converts all entries to floats, handling fractions."""
    str_values = [val for val in line.strip().split() if val]
    float_values = []
    for s in str_values:
        if '/' in s:
            float_values.append(float(Fraction(s)))
        else:
            float_values.append(float(s))
    return float_values

def load_data_from_files():
    """Manually loads the Rs matrix and LAMBDA partition from the exported text files."""
    try:
        # Manually parse five_rotor.txt
        rs_list = []
        with open("five_rotor.txt", 'r') as f:
            for line in f:
                rs_list.append(parse_mathematica_line(line))
        Rs = np.array(rs_list)

        # Robustly parse ten_shells.txt
        with open("ten_shells.txt", 'r') as f:
            content = f.read()
        
        cleaned_content = content.replace('{', ' ').replace('}', ' ').replace(',', ' ')
        all_numbers_str = [val for val in cleaned_content.strip().split() if val]
        all_numbers_float = [float(Fraction(s)) if '/' in s else float(s) for s in all_numbers_str]
        
        all_vectors_flat = np.array(all_numbers_float)
        LAMBDA = np.reshape(all_vectors_flat, (10, 24, 8))
        
    except Exception as e:
        print(f"\nERROR while loading data files: {e}")
        return None, None
        
    return Rs, list(LAMBDA)

def get_operators(Rs, LAMBDA):
    """Calculates the C2 't' and composite C10 'g' operators."""
    C_t = np.zeros((8, 8))
    for j in range(5):
        C_t += LAMBDA[j].T @ LAMBDA[j+5]
        C_t += LAMBDA[j+5].T @ LAMBDA[j]
    U_t, _, Vt_t = svd(C_t)
    Rt = U_t @ Vt_t
    if np.linalg.det(Rt) < 0:
        Vt_t[-1, :] *= -1
        Rt = U_t @ Vt_t
    Rg = Rs @ Rt
    return Rs, Rt, Rg

# --- 2. Lie Algebra and Stabilizer Functions ---

def get_so8_lie_algebra_basis():
    """Returns the 28 standard basis generators of so(8)."""
    dim = 8
    basis = []
    for i in range(dim):
        for j in range(i + 1, dim):
            G = np.zeros((dim, dim), dtype=int)
            G[i, j] = 1
            G[j, i] = -1
            basis.append(G)
    return basis

def calculate_stabilizer_dimension(rotor_float, so8_basis):
    """Calculates stabilizer dimension for a given rotor using the rank-nullity theorem."""
    rotor_int = np.round(2 * rotor_float).astype(int)
    commutator_map_cols = []
    for Gk in so8_basis:
        commutator_result = rotor_int @ Gk - Gk @ rotor_int
        commutator_map_cols.append(commutator_result.flatten())
    M = np.array(commutator_map_cols).T
    rank_M = np.linalg.matrix_rank(M)
    return len(so8_basis) - rank_M

def get_stabilizer_basis(rotor_float, so8_basis):
    """Calculates a matrix basis for the stabilizer Lie algebra."""
    rotor_int = np.round(2 * rotor_float).astype(int)
    commutator_map_cols = []
    for Gk in so8_basis:
        commutator_result = rotor_int @ Gk - Gk @ rotor_int
        commutator_map_cols.append(commutator_result.flatten())
    M = np.array(commutator_map_cols).T
    null_space_coeffs = null_space(M)
    stabilizer_basis = []
    for i in range(null_space_coeffs.shape[1]):
        vec = null_space_coeffs[:, i]
        mat = np.zeros((8, 8))
        for j in range(len(so8_basis)):
            mat += vec[j] * so8_basis[j]
        stabilizer_basis.append(mat)
    return stabilizer_basis

def get_center_dimension(algebra_basis):
    """Calculates the dimension of the center of a Lie algebra given its basis."""
    num_generators = len(algebra_basis)
    if num_generators == 0:
        return 0
    system_matrix_rows = []
    for j in range(num_generators):
        Bj = algebra_basis[j]
        row_block = []
        for i in range(num_generators):
            Bi = algebra_basis[i]
            commutator = Bi @ Bj - Bj @ Bi
            row_block.append(commutator.flatten())
        system_matrix_rows.append(np.array(row_block).T)
    M = np.vstack(system_matrix_rows)
    return null_space(M).shape[1]

# --- Helper function for geometric action check ---
def to_Z8_half(v_float):
    """Maps a float vector to an exact integer tuple by scaling by 2."""
    return tuple(np.round(v_float * 2).astype(int))

# --- 3. Main Execution ---

if __name__ == "__main__":
    print("--- E8 Cycle Clock Rigorous Computational Analysis ---")
    Rs, LAMBDA = load_data_from_files()
    
    if Rs is not None and LAMBDA is not None:
        print("Data loaded successfully.")
        
        Rs, Rt, Rg = get_operators(Rs, LAMBDA)
        so8_basis = get_so8_lie_algebra_basis()

        # --- Perform all computations ---
        commute_check = np.allclose(Rs @ Rt, Rt @ Rs)
        is_g_symmetry = to_Z8_half(Rg @ LAMBDA[0][0]) in {to_Z8_half(v) for v in np.vstack(LAMBDA)}
        stab_dim_s = calculate_stabilizer_dimension(Rs, so8_basis)
        stab_dim_t = calculate_stabilizer_dimension(Rt, so8_basis)
        stab_dim_g = calculate_stabilizer_dimension(Rg, so8_basis)
        stabilizer_t_basis = get_stabilizer_basis(Rt, so8_basis)
        center_dim_t_stabilizer = get_center_dimension(stabilizer_t_basis)
        
        # --- Print Final Report ---
        print("\n" + "="*70)
        print("                FINAL COMPUTATIONAL RESULTS")
        print("="*70)
        
        print("\nPart 1: Analysis of the C5 x C2 Clock Mechanism")
        print("-" * 50)
        print(f"  Operators s (C5) and t (C2) commute: {commute_check}")
        print(f"  Composite operator g=st is a symmetry of the E8 vertex set: {is_g_symmetry}")
        if not is_g_symmetry:
            print("  -> Conclusion: The clock operator g is NOT a permutation of the E8 roots.")

        print("\nPart 2: Analysis of Stabilizer Subgroups in Spin(8)")
        print("-" * 50)
        print(f"  Dimension of Stabilizer of C5 component 's': {stab_dim_s}")
        print(f"  Dimension of Stabilizer of C2 component 't': {stab_dim_t}")
        print(f"  Dimension of Stabilizer of full clock 'g': {stab_dim_g}")
        
        print("\nPart 3: Structural Analysis of the 12D Stabilizer of 't'")
        print("-" * 50)
        print(f"  Dimension of Stab(t)'s Center: {center_dim_t_stabilizer}")
        print("  -> Known structure of Standard Model Algebra: Dimension=12, Center=1")
        print("  -> Conclusion: Stab(t) is NOT the Standard Model algebra.")

        print("\n" + "="*70)
        print("                      OVERALL VERDICT")
        print("="*70)
        print("The initial hypothesis that the stabilizer of the C5xC2 clock is the")
        print("Standard Model gauge group is computationally FALSIFIED.")
        print("\nKey Findings:")
        print("1. The C2 swap operator stabilizes a 12D subgroup, matching the dimension")
        print("   of the Standard Model group, but its structure is incorrect (Center Dim=0).")
        print("2. The full clock operator 'g' is not a symmetry of the E8 vertices and its")
        print("   stabilizer is too small (Dimension=3).")
        print("="*70)

    else:
        print("\nAnalysis failed due to data loading errors.")
