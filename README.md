# Multi-Scale Measures in Variational Inference
&nbsp;

This repository contains code for calculating complexity- and emergence-related multi-scale measures during Variational Inference (VI, also called approximate Bayesian inference). 

## Approach

We use a hybrid approach—combining numerical and analytical methods—to simulate the evolution of Gaussian parameters during VI. These parameters serve as inputs for calculating:

- **Complexity measures** such as Integrated Information,
- **Emergence-related measures** such as Emergence Capacity (based on Partial and Integrated Information Decomposition).

Both types of measures are computed at each point in the evolutionary process.

## Current Status

**This work is ongoing and the code is not yet documented and tested for replication and wider use.** The project is led by Nadine Spychala in collaboration with Miguel Aguilera.

## Usage

The main script is `mec_var_inf_steady_state_param_sweep.ipynb`, which uses functions from `mec_var_inf.py` located in the `src` directory. Plotting is handled by `mec_var_inf_steady_state_param_sweep_plotting.ipynb`.

### Requirements

- Python with Matlab engine support
- Matlab (must be installed locally)

### Setup Note

This code is not yet packaged. You'll need to set local-specific directories at the top of both the script and module files.

## Publication

A publication titled "Exploring the Relation of Variational Inference and Multi-Scale Measures in a Minimal Model" is in preparation. **Corresponding updates to this repository will follow**.
