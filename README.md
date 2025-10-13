# Statistical Computing Examples

## Authors

- Kevin Machogu — BSCCS/2023/66850
- Sharlen Kinyua — BSCCS/2023/59148

This repo contains small examples in Julia, Python, and R under `BasicsOfData/`.

## Prerequisites

- Linux with bash
- Julia installed (verify with `julia --version`)
- Python 3.11+ installed
- R installed (optional for R example)

## Python

A virtual environment has been set up for this workspace and NumPy is pinned in `requirements.txt`.

Setup (optional if the `.venv` already exists):

```bash
# from repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the Python script:

```bash
# using the workspace venv directly
./.venv/bin/python BasicsOfData/math_operations.py

# or (after activating the venv)
python BasicsOfData/math_operations.py
```

## Julia

Run the Julia example:

```bash
julia BasicsOfData/math.operations.jl
```

Notes:

- Avoid using variable names that collide with Base (e.g., `names`, `sum`).
- Julia does not auto-recycle shorter arrays like R. Make broadcast shapes compatible.

## R

Run the R example (if you have R installed):

```bash
Rscript BasicsOfData/math_operations.R
```

## Troubleshooting

- If `numpy` is missing, ensure you are using the repo's virtual environment or reinstall:

  ```bash
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

- If Julia can't be found, install it via the official installer or your package manager.
