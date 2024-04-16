
# PINNICLE
Physics Informed Neural Networks for Ice and CLimatE

[![codecov](https://codecov.io/github/enigne/PINNICLE/graph/badge.svg?token=S7REK0IKJH)](https://codecov.io/github/enigne/PINNICLE)
[![Documentation Status](https://readthedocs.org/projects/pinnicle/badge/?version=latest)](https://pinnicle.readthedocs.io/en/latest/?badge=latest)

A Python library for solving ice sheet modeling problems using a unified framework with Physics Informed Neural Networks


---
**NOTE**

   This project is under active development.

---

**Documentation**: [pinnicle.readthedocs.io](https://pinnicle.readthedocs.io)

![](docs/images/pinn.png)

## Physics

- Momentum Conservation (stress balance):
  - Shelfy Stream Approximation (SSA)
  - MOno-Layer Higher-Order (MOLHO) ice flow model

- Mass Conservation (mass balance):
  - Thickness evolution

- Coupuling:
  - stress balance + mass balance


## Data format

- [ISSM](https://issm.jpl.nasa.gov) `model()` type, directly saved from ISSM by `saveasstruct(md, filename)`
- Scattered data


## More

- [Install and Setup](https://pinnicle.readthedocs.io/en/latest/installation.html#installation)
- [An example of stress balance](https://pinnicle.readthedocs.io/en/latest/examples/pinn.ssa.html)
- [An example of mass balance](https://pinnicle.readthedocs.io/en/latest/examples/pinn.mc.html)
- [An example of coupling stress balance with mass balance](https://pinnicle.readthedocs.io/en/latest/examples/pinn.mcssa.html)

