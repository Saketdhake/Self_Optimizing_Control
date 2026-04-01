# Installation

## Requirements

- Python ≥ 3.10
- A working [IPOPT](https://coin-or.github.io/Ipopt/) solver on your `PATH`

## Install from source

```bash
git clone https://github.com/Saketdhake/Self_Optimizing_Control.git
cd Self_Optimizing_Control
pip install -e .
```

This installs the `socf` package in editable (development) mode along with its dependencies: `numpy`, `pandas`, and `pyomo`.

## Installing IPOPT

SOCF requires the IPOPT nonlinear solver. The easiest way to install it:

**With Conda (recommended):**
```bash
conda install -c conda-forge ipopt
```

**On Ubuntu/Debian:**
```bash
sudo apt-get install coinor-libipopt-dev
```

After installation, verify IPOPT is available:
```python
import pyomo.environ as pyo
solver = pyo.SolverFactory("ipopt")
print(solver.available())  # Should print True
```

## Optional development dependencies

```bash
pip install -e ".[dev]"
```

This adds `pytest`, `ruff`, and `black` for testing and code formatting.