# Statistical Computing Examples

## Authors

- Kevin Machogu — BSCCS/2023/66850
- Sharlen Kinyua — BSCCS/2023/59148
- Sarah Githinji — BSCCS/2023/59148
- Edwin Meiteikini — BSCCS/2024/44160

This repo contains small examples in Julia, Python, and R under `BasicsOfData/`, and an R lesson on arrays and data frames in `Arrays and DataFrames/` that explains arrays, matrices, lists, and data frames with simple modeling and plots.
Additionally, a Python translation of the R lesson is provided to reproduce the same outputs.

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

### Python translation of "Arrays and Data Frames.r"

A Python version of the R script exists at `Arrays and DataFrames/arrays_and_data_frames.py`. It mirrors the printed outputs and plots from the original R lesson.

Run it with:

```bash
# using the workspace venv directly
./.venv/bin/python "Arrays and DataFrames/arrays_and_data_frames.py"

# or (after activating the venv)
python "Arrays and DataFrames/arrays_and_data_frames.py"
```

Notes:
- Plots are saved to the `plots/` directory:
  - `plots/allegheny_actual_vs_predicted.png` (only if the dataset download succeeds)
  - `plots/illiteracy_vs_frost.png`
- If the CSV download fails, the Pennsylvania regression and Allegheny plot steps will be skipped; the rest still runs.

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

### Arrays and Data Frames (R lesson)

This R script demonstrates core R data structures and simple modeling/plotting:

- Arrays and matrices: creation, indexing, row/column operations
- Matrix algebra: transpose, determinant, diagonal, inverse, solving linear systems
- Lists: heterogeneous containers, naming elements, accessing with `$`/`[[`]
- Data frames: creation, access by rows/columns, `with()`, summary statistics
- Modeling: simple linear regression on a housing dataset (Pennsylvania tracts)
- Eigen decomposition: extracting eigenvalues/vectors
- Plots: saves two PNGs to a `plots/` folder

Run the lesson script:

```bash
# from repo root; quotes handle spaces in path
Rscript "Arrays and DataFrames/Arrays and Data Frames.r"
```

Outputs:

- PNG files are saved to the `plots/` directory at the repo root:
  - `plots/allegheny_actual_vs_predicted.png`
  - `plots/illiteracy_vs_frost.png`

Notes:

- The Allegheny scatter plot requires downloading a CSV from CMU; if the network is unavailable, that plot will be skipped and the rest of the script still runs.
- The script is non-interactive: plots are written to files so it works in batch environments.
- The path contains spaces, so keep the quotes around the script path.

## Troubleshooting

- If `numpy` is missing, ensure you are using the repo's virtual environment or reinstall:

  ```bash
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

- If Julia can't be found, install it via the official installer or your package manager.
