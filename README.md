# E8 Cycle Clock Analysis

This repository contains the Python scripts and source data for the computational analysis presented in the paper *"Spin-Network Cycle Clocks in Spin(8): from Discrete Spinors to the Standard-Model Gauge Group"*.

The scripts in this repository perform a rigorous investigation of a procedural clock mechanism on the E8 root system. The analysis focuses on a "Fast Pointer, Slow Clicker" model driven by a C5 rotational symmetry (`s`) and a C4 bridging symmetry (`σ`), and calculates the algebraic structure of their stabilizer subgroups within Spin(8).

## Project Goal

The code computationally explores the symmetries of a 10-shell decomposition of the E8 root lattice. It is designed to be a self-contained demonstration of the following:

1.  **Clock Dynamics:** It verifies the existence and properties of a "Fast Pointer, Slow Clicker" mechanism that acts transitively on the 10 shells of the E8 lattice.
2.  **Stabilizer Calculation:** It calculates the dimensions of the stabilizer subgroups in Spin(8) for the clock's key components: the C5 "pointer" (`s`) and the C4 "clicker" (`σ`).
3.  **Standard Model Connection:** It analyzes these stabilizers to investigate a pathway to the Standard Model gauge group, focusing on the discovery of a 16-dimensional stabilizer subgroup isomorphic to `U(4)`.

## Repository Contents

* **`tests/`**: A directory containing various Python scripts used during the research and development process.
* **`e8_clock_analysis.py`**: The original script for analyzing the simple (and ultimately flawed) C5 and C2 clock models.
* **`e8_clock_analysis_c4.py`**: The script that performs the stabilizer analysis of the final working model's components: the C5 pointer (`s`) and the C4 clicker (`σ`).
* **`e8_clock_analysis_c4_sm.py`**: The script used to extract the 16 basis generators of the `Stab(σ)` stabilizer group.
* **`e8_clock_fast_slow_pointers.py` / `..._check.py`**: Scripts used to verify the "Fast Pointer, Slow Clicker" procedural clock dynamics.
* **`LICENSE`**: The MIT License file.
* **`README.md`**: This file.


## Requirements

The scripts require the following Python libraries:

  * NumPy
  * SciPy

You can install them using pip:

```bash
pip install numpy scipy
```

## Usage

To run the primary analysis, navigate to the scripts directory and execute the main analysis file.

```bash
python3 main_analysis.script.py
```