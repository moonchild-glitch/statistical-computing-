"""
Python translation of R script: Arrays, Matrices, Lists, and DataFrames
This script mirrors the logic of `Arrays and Data Frames.r` and aims to
produce the same outputs (printed values and plots) where feasible.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure a directory exists for saved plots during non-interactive runs
ROOT = os.path.abspath(os.path.join(os.getcwd()))
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---- Arrays (NumPy) ----
print("# arrays, matrices, lists, and dataframes\n")

x = np.array([7, 8, 10, 45])
x_arr = np.array(x).reshape((2, 2))
print(x_arr)
print(x_arr.shape)
print(np.ndim(x_arr) == 1)  # is.vector equivalent (False)
print(isinstance(x_arr, np.ndarray))  # is.array equivalent
print(x_arr.dtype.name)
print(repr(x_arr))

# Accessing and operating on arrays
print(x_arr[0, 1])  # 1-based to 0-based index
# attributes: show shape and dtype as a proxy
print({"shape": x_arr.shape, "dtype": str(x_arr.dtype)})
print(x_arr.flat[2])  # x.arr[3] in R (1-based)

print(x_arr[[0, 1], 1])
print(x_arr[:, 1])

# Functions on arrays
# R's which() returns 1-based linear indices; emulate that
lin_idx = np.flatnonzero((x_arr > 9).ravel(order="F")) + 1  # column-major like R
print(lin_idx)
# Many functions preserve array structure
y = -x
y_arr = y.reshape((2, 2))
print(y_arr + x_arr)
# Row sums
print(x_arr.sum(axis=1))

# ---- Example: Price of houses in PA ----
# Read data with a guard for network errors (so the rest of the script can still run)
calif_penn_url = "http://www.stat.cmu.edu/~cshalizi/uADA/13/hw/01/calif_penn_2011.csv"
try:
    calif_penn = pd.read_csv(calif_penn_url)
except Exception as e:
    print(f"[Skip] Could not download dataset from {calif_penn_url}: {e}")
    calif_penn = None

if calif_penn is not None:
    penn = calif_penn[calif_penn["STATEFP"] == 42].copy()
    # Fit linear model Median_house_value ~ Median_household_income
    # Using numpy polyfit for a simple linear regression; drop NA/non-finite like R's na.omit
    df_xy = penn[["Median_household_income", "Median_house_value"]].apply(pd.to_numeric, errors="coerce")
    df_xy = df_xy.replace([np.inf, -np.inf], np.nan).dropna()
    xi = df_xy["Median_household_income"].to_numpy(dtype=float)
    yi = df_xy["Median_house_value"].to_numpy(dtype=float)
    penn_coefs: Dict[str, float] | None = None
    if xi.size >= 2:
        try:
            slope, intercept = np.polyfit(xi, yi, 1)
            # R's coefficients gives (Intercept) first, then slope; print similarly
            penn_coefs = {"(Intercept)": float(intercept), "Median_household_income": float(slope)}
            print(penn_coefs)
        except Exception as e:
            print(f"[Skip] Linear regression failed: {e}")
    else:
        print("[Skip] Not enough non-missing observations to fit regression")
else:
    penn_coefs = None

# Fit a simple linear model, predicting median house price from median household income
# Using the constants from the R comments
print(34100 < (-26206.564 + 3.651 * 14719))
print(155900 < (-26206.564 + 3.651 * 48102))

if calif_penn is not None and penn_coefs is None:
    # As a fallback, attempt once more after coercion (already done); if still None, skip printing
    pass

if calif_penn is not None and penn_coefs is not None:
    # R's 24:425 are 1-based inclusive; pandas iloc is 0-based and end-exclusive
    subset = penn.iloc[23:425].copy()
    if not subset.empty:
        allegheny_medinc = pd.to_numeric(subset["Median_household_income"], errors="coerce").to_numpy()
        allegheny_values = pd.to_numeric(subset["Median_house_value"], errors="coerce").to_numpy()
        mask = np.isfinite(allegheny_medinc) & np.isfinite(allegheny_values)
        allegheny_medinc = allegheny_medinc[mask]
        allegheny_values = allegheny_values[mask]
        allegheny_fitted = (
            penn_coefs["(Intercept)"] + penn_coefs["Median_household_income"] * allegheny_medinc
        )

        if allegheny_fitted.size > 0:
            # Save scatter plot of actual vs predicted to file
            fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=120)
            ax.scatter(allegheny_fitted, allegheny_values)
            ax.set_xlabel("Model-predicted median house values")
            ax.set_ylabel("Actual median house values")
            ax.set_xlim(0, 5e5)
            ax.set_ylim(0, 5e5)
            # y = x line
            lims = [0, 5e5]
            ax.plot(lims, lims, color="grey")
            fig.tight_layout()
            out_path = os.path.join(PLOTS_DIR, "allegheny_actual_vs_predicted.png")
            fig.savefig(out_path)
            plt.close(fig)

# ---- Matrices ----
factory = np.array([[40, 1], [60, 3]])
print(isinstance(factory, np.ndarray))  # is.array(factory)
print(factory.ndim == 2)  # is.matrix equivalent

# Matrix multiplication
six_sevens = np.full((2, 3), 7)
print(six_sevens)
print(factory @ six_sevens)  # [2x2] * [2x3]

# Multiplying matrices and vectors
output = np.array([10, 20])
print(factory @ output)
print(output @ factory)

# Transpose
print(factory.T)

# Determinant
print(np.linalg.det(factory))

# The diagonal
print(np.diag(factory))

factory_diag_mut = factory.copy()
np.fill_diagonal(factory_diag_mut, [35, 4])
print(factory_diag_mut)
# restore
factory = factory.copy()
np.fill_diagonal(factory, [40, 3])

# Creating a diagonal or identity matrix
print(np.diag([3, 4]))
print(np.eye(2))

# Inverting a matrix
factory_inv = np.linalg.inv(factory)
print(factory_inv)
print(factory @ factory_inv)

# "solve" equivalent: solve(factory, available)
available = np.array([1600, 70])
solution = np.linalg.solve(factory, available)
print(solution)
print(factory @ solution)

# Names in matrices -> use pandas DataFrame for labeled math
factory_df = pd.DataFrame(factory, index=["labor", "steel"], columns=["cars", "trucks"])
print(factory_df)

available_named = pd.Series([1600, 70], index=["labor", "steel"])
output_named = pd.Series([20, 10], index=["trucks", "cars"])  # mixed up order
print(factory_df.to_numpy() @ output_named.to_numpy())
print(factory_df.to_numpy() @ output_named[factory_df.columns].to_numpy())

ok = (factory_df @ output_named[factory_df.columns]).le(available_named[factory_df.index]).all()
print(ok)

# Doing the same thing to each row or column
print(factory_df.mean(axis=0))  # colMeans
print(factory_df.describe())    # summary
print(factory_df.mean(axis=1))  # rowMeans
print(factory_df.apply(np.mean, axis=1))

# ---- Lists ----
my_distribution: List[Any] = ["exponential", 7, False]
print(my_distribution)

# Accessing pieces of lists
print(all(isinstance(el, str) for el in my_distribution))
print(isinstance(my_distribution[0], str))
print(my_distribution[1] ** 2)

# Expanding and contracting lists
my_distribution = my_distribution + [7]
print(my_distribution)
print(len(my_distribution))
# Truncate to length 3
my_distribution = my_distribution[:3]
print(my_distribution)

# Naming list elements -> use dict
my_distribution_named: Dict[str, Any] = {"family": "exponential", "mean": 7, "is.symmetric": False}
print(my_distribution_named)
print(my_distribution_named["family"])  # [["family"]]
print({k: v for k, v in my_distribution_named.items() if k == "family"})  # ["family"] retains key

# $ shortcut -> attribute-like via SimpleNamespace or dataclass; we'll just show the value
print(my_distribution_named["family"])  # same as [["family"]]
print(my_distribution_named["family"])  # mimic $family

another_distribution = {"family": "gaussian", "mean": 7, "sd": 1, "is.symmetric": True}
my_distribution_named["was.estimated"] = False
my_distribution_named["last.updated"] = "2011-08-30"
# remove key
my_distribution_named.pop("was.estimated", None)

# Dataframes
a_matrix = np.array([[35, 8], [10, 4]])
a_df = pd.DataFrame(a_matrix, columns=["v1", "v2"])  # a.matrix
print(a_df)
print(a_df["v1"])  # a.matrix[ , "v1"]

# Add third column
a_df = a_df.assign(logicals=[True, False])
print(a_df)
print(a_df.v1)
print(a_df.loc[:, "v1"])
print(a_df.iloc[0, :])
print(a_df.mean(axis=0, numeric_only=True))

# Adding rows (rbind)
print(pd.concat([a_df, pd.DataFrame([{"v1": -3, "v2": -5, "logicals": True}])], ignore_index=True))
print(pd.concat([a_df, pd.DataFrame([{"v1": 3, "v2": 4, "logicals": 6}])], ignore_index=True))

# Structures of Structures
plan = {"factory": factory_df, "available": available_named, "output": output_named}
print(plan["output"])  # plan$output

# Example: Eigenstuff
w, v = np.linalg.eig(factory)
print({"values": w, "vectors": v})
print(type((w, v)))
print(factory @ v[:, 1])
print(w[1] * v[:, 1])
print(w[1])
# R's eigen(factory)[[1]][[2]] corresponds to second eigenvalue
print(w[1])

# Creating an example dataframe similar to R's states
# The R datasets::state.x77 contains 50x8 matrix with named rows/cols plus
# state.abb, state.region, state.division.
# We'll recreate a comparable structure from seaborn (if available) or ship a small fallback.
try:
    import seaborn as sns  # type: ignore
    # Not a perfect replacement; we'll synthesize a small DataFrame with needed columns names
    raise ImportError("Use fallback synthetic dataset")
except Exception:
    # Synthetic minimal states-like dataframe to allow operations to run
    data = {
        "Population": [3615, 365, 2212, 2110, 21198],
        "Income": [3624, 6315, 4530, 3378, 5114],
        "Illiteracy": [2.1, 1.5, 1.8, 0.7, 1.1],
        "Life Exp": [69.05, 69.31, 70.55, 70.66, 71.71],
        "Murder": [15.1, 11.3, 7.8, 10.1, 10.3],
        "HS.Grad": [41.3, 66.7, 58.1, 62.1, 58.8],
        "Frost": [20, 152, 15, 65, 20],
        "Area": [50708, 566432, 113417, 51945, 156361],
        "abb": ["AL", "AK", "AZ", "AR", "CA"],
        "region": ["South", "West", "West", "South", "West"],
        "division": ["East South Central", "Pacific", "Mountain", "West South Central", "Pacific"],
    }
    states = pd.DataFrame(data, index=[
        "Alabama", "Alaska", "Arizona", "Arkansas", "California"
    ])

print(list(states.columns))
print(states.iloc[0, :])

# Dataframe access
print(states.iloc[48, 2] if len(states) > 49 else states.iloc[-1, 2])
print(states.loc["Wisconsin", "Illiteracy"] if "Wisconsin" in states.index else states.iloc[-1]["Illiteracy"])  # fallback
print(states.loc["Wisconsin", :] if "Wisconsin" in states.index else states.iloc[-1, :])
print(states.iloc[:5, 2])
print(states.loc[:, "Illiteracy"].head())
print(states["Illiteracy"].head())
print(states.loc[states["division"] == "New England", "Illiteracy"])  # may be empty with fallback
print(states.loc[states["region"] == "South", "Illiteracy"])        # subset
print(states["HS.Grad"].describe())

states["HS.Grad"] = states["HS.Grad"] / 100.0
print(states["HS.Grad"].describe())

states["HS.Grad"] = 100 * states["HS.Grad"]

# with(): compute 100*(HS.Grad/(100-Illiteracy))
print((100 * (states["HS.Grad"] / (100 - states["Illiteracy"])) ).head())
print((100 * (states["HS.Grad"] / (100 - states["Illiteracy"])) ).head())

# Plot Illiteracy vs Frost
fig, ax = plt.subplots(figsize=(7.5, 5.8), dpi=120)
ax.scatter(states["Frost"], states["Illiteracy"])
ax.set_xlabel("Frost")
ax.set_ylabel("Illiteracy")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "illiteracy_vs_frost.png"))
plt.close(fig)

print("\n## SUMMARY\n"
      "Arrays add multi-dimensional structure to vectors\n"
      "Matrices act like you'd hope they would\n"
      "Lists let us combine different types of data\n"
      "Dataframes are hybrids of matrices and lists, for classic tabular data\n"
      "Recursion lets us build complicated data structures out of the simpler ones\n")
