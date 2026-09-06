"""RUL-CMAPSS: Remaining Useful Life prediction on the NASA C-MAPSS dataset.

Pipeline: data loading -> per-engine rolling features -> LightGBM point
model -> split-conformal prediction intervals (MAPIE), evaluated with RMSE
and the NASA asymmetric scoring function, and served via FastAPI.
"""

__version__ = "0.1.0"
