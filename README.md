
# Data-Driven Reduced-Order Models for Port-Hamiltonian Systems with Operator Inference

This repository contains the numerical implementation associated with the paper

**Data-driven reduced-order models for port-Hamiltonian systems with operator inference**  
Yuwei Geng, Lili Ju, Boris Kramer, and Zhu Wang  
*Computer Methods in Applied Mechanics and Engineering*, 442 (2025), 118042.

The repository contains the implementation of the linear mass-spring-damper example and includes the following reduced-order modeling approaches:

- Structure-Preserving Galerkin ROM (`sp`)
- pH-OpInf-R (`PH-OpInf_R`)
- pH-OpInf-W (`PH-OpInf_W`)

## Requirements

The code requires the Python package cvxpy


## Files

- `FOM_solution.py`  
  Full-order model (FOM) simulation.

- `ROM_solution.py`  
  Reduced-order model (ROM) simulation.

- `matrix.py`  
  Construction of the mass-spring system matrices.

- `utils.py`  
  Utility functions, including numerical differentiation and error computation.

- `Op-Inf.py`  
  Main script for running the three reduced-order modeling methods:
  - Structure-Preserving Projection (`sp`)
  - PH-OpInf-R (`PH-OpInf_R`)
  - PH-OpInf-W (`PH-OpInf_W`)

- `OpInf_R_test.py`  
  Test script for studying the effect of the regularization parameter in PH-OpInf-R.

- `OpInf_w_test.py`  
  Test script for studying the effect of the weighting parameter in PH-OpInf-W.
