##########################################
# Special Concentric Braced Frame (SCBF) Module
# Written by: Hossein Karagah
# Date: 2025-11-14
# Description: This module provides classes and functions to handle snow load calculations per ASCE 7.
##########################################

from __future__ import annotations
from typing import Mapping, Optional, Protocol, Dict, Tuple, List

import os
import sys
sys.path.append(os.path.abspath("../../.."))
from assets.modules.shapes import AISCShape
from assets.modules.materials import ASTMSteel

import numpy as np
from math import pi, radians, cos, sin, sqrt


def snow_density(Pg: Optional[float | np.ndarray]) -> np.ndarray:
    if isinstance(Pg, float):
        Pg = np.array(Pg)
        
    gamma = np.minimum(0.13 * Pg + 14, 30)  # lb/ft^3
    
    if gamma.ndim == 1 and gamma.size == 1:
        gamma = float(gamma)
        
    return gamma


def asce_7_16_drift_height(
    Is: Optional[float | np.ndarray],
    Pg: Optional[float | np.ndarray],
    lu: Optional[float | np.ndarray],
) -> np.ndarray:
    # Convert to arrays
    Is = np.array(Is, dtype=float).reshape(-1, 1, 1)  # (n_Is, 1, 1)
    Pg = np.array(Pg, dtype=float).reshape(1, -1, 1)  # (1, n_Pg, 1)
    lu = np.array(lu, dtype=float).reshape(1, 1, -1)  # (1, 1, n_lu)

    hd = (0.43 * np.power(lu, 1/3) * np.power(Pg + 10, 1/4) - 1.5) * np.sqrt(Is)

    # Optional: if it’s truly scalar, return scalar
    if hd.ndim == 1 and hd.size == 1:
        hd = float(hd)

    return hd


def asce_7_22_drift_height(
    Pg: Optional[float | np.ndarray], 
    lu: Optional[float | np.ndarray], 
    W2: Optional[float | np.ndarray]
) -> np.ndarray:
    # Convert to arrays
    Pg = np.array(Pg, dtype=float).reshape(-1, 1, 1)  # (n_Pg, 1, 1)
    lu = np.array(lu, dtype=float).reshape(1, -1, 1)  # (1, n_lu, 1)
    W2 = np.array(W2, dtype=float).reshape(1, 1, -1)  # (1, 1, n_W2)
        
    gamma = snow_density(Pg).reshape(Pg.shape)
    
    hd = 1.5 * np.sqrt(np.power(Pg, 0.74) * np.power(lu, 0.7) * np.power(W2, 1.7) / gamma)

    if hd.ndim == 1 and hd.size == 1:
        hd = float(hd)
        
    return hd








# Main function =========================================================
def main():
    pass
    
    
if __name__ == "__main__":
    main()