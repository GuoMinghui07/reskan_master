## Overview

Extracting the governing equation for complex systems is a critical task in scientific research. Previous research often focuses on the static properties of governing equations, while real-world dynamics involve complex, evolving factors that influence system behavior. This work proposes a novel approach that integrates single-layer Kolmogorov-Arnold networks (KAN) in the downstream operations of physics-informed neural networks (PINNs), combined with an alternating training strategy using sparse regression algorithms.

Unlike traditional methods, this approach relies solely on sparse data, without requiring any prior knowledge, to reconstruct the precise form of partial differential equations (PDEs) and simultaneously identify variable-coefficient functions that depend on single variables. By symbolizing the spline functions within the KAN layer, this method can also derive the expressions of these coefficient functions and reveal key parameters with real physical significance. Numerical experiments demonstrate the effectiveness and robustness of this method in handling complex systems and data with varying levels of sparsity and noise, providing a new solution for reconstructing and analyzing dynamic equations.

## Headware Requirements
NVIDIA GeForce RTX 4090 24GB


## Installation

To set up the environment, ensure that you have Python 3.9.7 or higher installed, and then use the following command to install the required libraries:

```bash
pip install -r requirements.txt
```

