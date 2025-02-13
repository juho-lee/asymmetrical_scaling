# Asymmetrical Scaling

This repository contains the code to reproduce the experiments and figures from the paper:
**"Over-parameterised Shallow Neural Networks with Asymmetrical Node Scaling: Global Convergence Guarantees and Feature Learning"**.


## Code Overview

### `ffnn.py`
This file implements a feedforward neural network (FFNN) with asymmetrical node scaling:
- `Scaling`: A custom module that scales input activations.
- `ScaledFCLayer`: A fully connected layer with asymmetrical scaling.
- `FFNN`: Feedforward neural network which (or without) scaled layers.

### `sampling_utils.py`
Provides utilities for sampling different distributions used in the initialization and regularization of the network:
- **GammaInc**: Implements the incomplete gamma function with autograd support.
- **Sample Finite GBFRY, GGP, Stable**: Implements sampling functions for various statistical distributions.
- **Lam Samplers**: Different variance initializations for network training, including Horseshoe, Beta, and GBFRY distributions.

## Running Experiments

Each subfolder (`cifar10/`, `mnist/`, `regression/`, `simulations/`) contains scripts to run specific experiments:
- `run.py`: Executes a single experiment.
- `script.sh`: Batch execution of multiple experiments.
- `visualize.ipynb`: Jupyter notebook to process and visualize results. Reproduces the plots of the paper.



## License
This project is licensed under the terms of the MIT License.


