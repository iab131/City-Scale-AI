# City-Scale Traffic Forecasting with Graph Fourier Transform (GFT)

This project implements a traffic forecasting pipeline on the **METR-LA** dataset using the **Graph Fourier Transform (GFT)** as the spatial representation step.

The idea is:

1. Load traffic signals from METR-LA
2. Build the road sensor graph from the provided adjacency matrix
3. Compute the normalized graph Laplacian
4. Perform eigendecomposition to obtain the graph Fourier basis
5. Transform traffic signals into the spectral domain
6. Train a temporal model on spectral coefficients
7. Reconstruct predictions back to sensor space with inverse GFT

---

## Project Goal

Standard graph models often rely on local message passing.  
This project instead moves node signals into the **spectral domain**, where traffic can be represented as global graph modes.

This makes it easier to model:

- city-wide congestion patterns
- long-range dependencies
- smooth vs. abrupt traffic variations

---

## Dataset

We use the **METR-LA** traffic forecasting dataset.

Expected files:

```text
data/metr_la/
├── metr-la.h5
├── adj_METR-LA.pkl