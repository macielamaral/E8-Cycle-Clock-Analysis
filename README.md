# E8 Cycle Clock Analysis

This repository contains the Python script and source data for the computational analysis presented in the paper *"Discrete Spinors, the $C_{5}\times C_{2}$ Cycle Clock, and the Stabilizer Group in $\mathrm{Spin}(8)$"*.

The primary script, `e8_clock_analysis.py`, performs a rigorous investigation of a `C₅ × C₂` symmetry structure within the E8 root system, analyzing its geometric action and the algebraic structure of its stabilizer subgroups.

## Project Goal

The code computationally verifies several key theorems about the E8 root lattice and its symmetries. It is designed to be a self-contained proof that:

1.  Verifies the existence and properties of a `C₅ × C₂` "Cycle Clock" that acts on 10 shells of the E8 lattice.
2.  Calculates the dimensions of the stabilizer subgroups in `Spin(8)` for the `C₅` pointer (`s`), the `C₂` pointer (`t`), and the composite clock (`g`).
3.  Analyzes the structure of the resulting 12-dimensional stabilizer of the `C₂` component by calculating the dimension of its center, allowing for a direct comparison with the Standard Model gauge group.

## Repository Contents

* `e8_clock_analysis.py`: The main Python script to run the full analysis.
* `five_rotor.txt`: The source data for the 8x8 `C₅` rotation matrix (`s`).
* `ten_shells.txt`: The source data for the partition of the 240 E8 roots into 10 D₄ shells, formatted as exported from Mathematica.
* `LICENSE`: The MIT License file.
* `README.md`: This file.

## Requirements

The script requires the following Python libraries:

* NumPy
* SciPy

You can install them using pip:
```bash
pip install numpy scipy
