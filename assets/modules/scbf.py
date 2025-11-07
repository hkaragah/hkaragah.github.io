##########################################
# Special Concentric Braced Frame (SCBF) Module
# Written by: Hossein Karagah
# Date: 2025-11-04
# Description: This module provides classes and functions to handle SCBF analysis and design.
##########################################

from __future__ import annotations
from typing import Mapping, Optional, Protocol, Dict, Tuple, List

import os
import sys
sys.path.append(os.path.abspath("../../.."))
from assets.modules.shapes import AISCShape
from assets.modules.materials import ASTMSteel

import numpy as np
from math import pi, radians, cos, sin


   

class Brace:
    """
    Brace member for OCBF/SCBF checks.
    - Tension positive, compression negative.
    """
    
    E = 29000  # Steel modulus of elasticity in ksi.
    PHI_T = 0.9 # LRFD phi in tension.
    PHI_C = 0.9 # LRFD phi in compression.
    
    def __init__(self, label: str, length: float, mat: ASTMSteel, K: float=1.0):
        """This class represents a brace member in a OCBF or SCBF.

        Args:
            label (str): The label of the brace section.
            length (float): The length of the brace.
            mat (ASTMSteel): The material properties of the brace.
        """
        self.label = label
        self._props = AISCShape(label).props()
        self.length = float(length)
        self.Fy = float(mat.fy)
        self.Ry = float(mat.Ry)
        self.K = float(K)
            
    @property
    def A(self):
        "Area, in^2."
        return self._props['A']
    
    @property
    def rx(self):
        "Radius of gyration about x, in."
        return self._props['rx']
    
    @property
    def ry(self):
        "Radius of gyration about y, in."
        return self._props['ry']
    
    @property
    def slenderness_ratio(self):
        "KL/r about weak axis. Assumes pinned-pinned unless K set otherwise."
        return self.K * self.length / min(self.rx, self.ry)
    
    def Fcr(self, Fy: Optional[float] = None):
        """
        AISC column strength stress F_cr (ksi), Eqn E3-2 / E3-3 style:
            Fe = pi^2*E / (KL/r)^2
            If Fy/Fe <= 2.25: Fcr = 0.658^(Fy/Fe) * Fy
            Else:             Fcr = 0.877 * Fe
        """
        Fy = self.Fy if Fy is None else float(Fy)
        Fe = (pi ** 2 * self.E) / (self.slenderness_ratio ** 2)
        if Fy/Fe <= 2.25:
            return Fy * (0.658 ** (Fy / Fe))
        else:
            return 0.877 * Fe
    
    @property
    def Tn(self):
        """Nominal tensile yield strength (no phi), kips."""
        return self.A * self.Fy
    
    @property
    def Pn(self):
        """Nominal compressive strength (no phi), kips."""
        return self.A * self.Fcr()
    
    @property
    def phiTn(self) -> float:
        """Design tensile strength (LRFD), kips."""
        return self.PHI_T * self.Tn

    @property
    def phiPn(self) -> float:
        """Design compressive strength (LRFD), kips."""
        return self.PHI_C * self.Pn
    
    @property
    def Texp(self):
        "Expected tensile strength in kips."
        return self.Ry * self.A * self.Fy
    
    @property
    def Cexp(self):
        "Expected compressive strength in kips."
        Fne = self.Fcr(self.Ry * self.Fy)
        return min(self.Texp, self.A * Fne / 0.877)
    
    @property
    def Cpost(self):
        "Post-buckling compressive strength in kips."
        return 0.3 * self.Cexp


class GussetPlate:
    def __init__(self, Lg: float, h: float, loc: Tuple[float, float]=(0.0, 0.0)):
        """This class defines the geometry and material of a gusset plate.

        Args:
            Lg (float): Length of gusset plate along the connected beam (inches).
            h (float): Height of gusset plate (inches).
            loc (tuple[float, float], optional): Location of the center of gusset-to-beam interface relative to work point (w.p.) in inches. Defaults to (0.0, 0.0). Right and up positive.
            
                loc=(Delta, eb) where: \n
                Delta: Horizontal distance from w.p. to center of gusset-to-beam interface (inches). Right positive. \n
                eb: Vertical distance from w.p. to center of gusset-to-beam interface (inches). Up positive.
        """
        self.Lg = Lg
        self.h = h
        self.loc = loc


def sum_horizontal_and_vertical_forces(forces: Dict[str, Dict[str, Force]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Sum horizontal and vertical forces from top and bottom chevron braces.

    Args:
        forces (dict): A dictionary containing Force objects for 'left' and 'right' braces connected to 'top' and 'bottom' gussets.
            {
                'top': {'left': Force, 'right': Force}, \n
                'bottom': {'left': Force, 'right': Force}
            }

    Returns:
        tuple: Two dictionaries containing summed horizontal (HT) and vertical (VT) forces for 'top' and 'bottom' gussets.
        
            (
                HT = {'top': PLt.X + PRt.X, 'bottom': PLb.X + PRb.X}, \n
                VT = {'top': PLt.Y + PRt.Y, 'bottom': PLb.Y + PRb.Y}
            )
    """
    _validate_gusset_forces(forces)
    
    HT, VT = {}, {}
    for key in forces.keys():
        Pl = forces[key]['left']
        Pr = forces[key]['right']
        HT[key] = Pl.X + Pr.X
        VT[key] = Pl.Y + Pr.Y
    
    return HT, VT

    
def chevron_gusset_forces_at_beam_interface(gussets: Dict[str, GussetPlate], forces: Dict[str, Dict[str, Force]]) -> Dict[str, Tuple[float, float, float]]:
    """
    Calculate the forces along the top and bottom gusset-to-beam interface in a chevron brace gusset plate. See Fortney & Thornton (2015) and (2017) for reference.

    Args:
        gussets (dict): A dictionary containing GussetPlate objects for 'top' and 'bottom' gussets.
        forces (dict): A dictionary containing Force objects for 'left' and 'right' braces connected to 'top' and 'bottom' gussets.
            {
                'top': {'left': Force, 'right': Force}, \n
                'bottom': {'left': Force, 'right': Force}
            }

    Returns:
        dict: A dictionary with keys 'top' and 'bottom', each containing a tuple of (H, V, M) forces at the gusset-to-beam interface.
        
            {
                'top': (H, V, M), \n
                'bottom': (H, V, M)
            }
    """
    _validate_gussets(gussets)
    _validate_gusset_forces(forces)
    
    # Initialize dictionary to hold gusset-to-beam interface H, V, M forces
    HVM = {}
    for key in gussets.keys():
        # Vertical distance from work point to center of gusset-to-beam interface
        eb = gussets[key].loc[1]
        
        # Horizontal distance from work point to center of gusset-to-beam interface
        Delta = gussets[key].loc[0]

        # Left and right brace forces
        Pl = forces[key]['left']
        Pr = forces[key]['right']

        # Calculate horizontal (H) and vertical (V) forces at gusset interface
        H = -(Pl.X + Pr.X)
        V = -(Pl.Y + Pr.Y)
        
        # Calculate moment at gusset-to-beam interface due to horizontal and vertical eccentricities
        M = (Pl.Y + Pr.Y) * Delta + (Pl.X + Pr.X) * eb
            
        HVM[key] = (H, V, M)  

    return HVM


def uniform_moment_due2_horizontal_force(gussets: Dict[str, GussetPlate], forces: Dict[str, Dict[str, Force]]) -> float:
    """
    Calculate the distributed uniform moment due to eccentricity in horizontal forces of the top and bottom gusset plates.

    Args:
        gussets (dict): A dictionary containing GussetPlate objects for 'top' and 'bottom' gussets.
        forces (dict): A dictionary containing Force objects for 'left' and 'right' braces connected to 'top' and 'bottom' gussets.
            {
                'top': {'left': Force, 'right': Force}, \n
                'bottom': {'left': Force, 'right': Force}
            }

    Returns:
        dict: A dictionary with keys 'top' and 'bottom', each containing the uniform moment (kip-in/in) due to horizontal forces eccentricities relative to w.p.
        
            {
                'top': M_top, \n
                'bottom': M_bottom
            }
    """
    _validate_gussets(gussets)
    _validate_gusset_forces(forces)
    
    sum_H = {}
    M = {}
    for key in gussets.keys():
        Pl = forces[key]['left']
        Pr = forces[key]['right']
        sum_H[key] = Pl.X + Pr.X
        
        # Vertical distance from work point to center of gusset-to-beam interface. Up positive.
        eb = gussets[key].loc[1]  # inches
        
        # Length of gusset-to-beam interface
        Lg = gussets[key].Lg      # inches
        
        M[key] = sum_H[key] * eb / Lg  # kip-in/in
    
    return M


def uniform_load_due2_vertical_force_and_moment(gussets: Dict[str, GussetPlate], forces: Dict[str, Dict[str, Force]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate the distributed uniform load due to vertical forces and moments in the gusset plate.

    Args:
        gusset (GussetPlate): The gusset plate object.
        forces (dict): A dictionary containing Force objects for 'top_left', 'top_right', 'bottom_left', and 'bottom_right' braces.

    Returns:
        dict: A dictionary with keys 'top' and 'bottom', each containing another dictionary with keys 'left' and 'right' representing the uniform loads (kips/in) on the left and right half of the gusset plate.
        
            {
                'top': {'left': Wl_top, 'right': Wr_top}, \n
                'bottom': {'left': Wl_bottom, 'right': Wr_bottom}
            }
    """
    _validate_gussets(gussets)
    _, VT = sum_horizontal_and_vertical_forces(forces)
    F_gusset= chevron_gusset_forces_at_beam_interface(gussets, forces)

    # Calculate uniform loads due to vertical force and moment couples at gusset-to-beam interface
    uniform_loads = {}
    for key in gussets.keys():
        # Moment at gusset-to-beam interface
        Ma = F_gusset[key][2] # kip-in
        
        # Length of gusset-to-beam interface
        Lg = gussets[key].Lg   # inches
        
        uniform_loads[key] = {
            'left': -4 * Ma / (Lg ** 2) + VT[key] / Lg,  # kips/in
            'right': 4 * Ma / (Lg ** 2) + VT[key] / Lg   # kips/in
        }
    
    return uniform_loads


def beam_reactions_due2_chevron_effect(L: float, x_wp: float, gussets: Dict[str, GussetPlate], forces: Dict[str, Dict[str, Force]]) -> Tuple[float, float]:
    """
    Calculate the beam reactions due to chevron brace forces acting on the gusset plate.

    Args:
        L (float): Length of the beam span (inches).
        x_wp (float): Distance from the left beam support to the work point (inches).
        gussets (dict): A dictionary containing GussetPlate objects for 'top' and 'bottom' gussets.
        forces (dict): A dictionary containing Force objects for 'left' and 'right' braces connected to 'top' and 'bottom' gussets.
            {
                'top': {'left': Force, 'right': Force}, \n
                'bottom': {'left': Force, 'right': Force}
            }

    Returns:
        tuple: A tuple containing the left and right beam reactions (R_left, R_right) in kips.
    """
    _validate_gussets(gussets)
    _validate_gusset_forces(forces)
    
    # Total vertical force from all braces
    _, VT = sum_horizontal_and_vertical_forces(forces)
    
    R_left, R_right = 0.0, 0.0
    
    for key in gussets.keys():
        # Horizontal distance from work point to center of gusset-to-beam interface
        Delta = gussets[key].loc[0]  # inches
        
        R_right += -VT[key] * (x_wp + Delta) / L  # kips
        R_left += -VT[key] * (L - x_wp - Delta) / L  # kips
    
    return R_left, R_right


def beam_internal_shear_due2_chevron_effect(x: np.ndarray , x_wp: float, gussets: Dict[str, GussetPlate], forces: Dict[str, Dict[str, Force]]) -> np.ndarray:
    """
    Internal shear V(x) (kips) along a simply supported beam due to chevron-brace
    forces resolved as equivalent uniform loads on the gusset footprint.

    Sign convention:
      - Downward loads positive (+), reactions Rl, Rr upward.
      - Shear V(x) is positive when the left segment tends to rotate clockwise
        (standard structural sign; adjust if your codebase uses the opposite).

    Args:
        x: 1D array of x-coordinates along the beam (inches).
        x_wp: Work point location from left support (inches).
        gussets (dict): A dictionary containing GussetPlate objects for 'top' and 'bottom' gussets.
        forces: dict of Force objects used by the helper functions.

    Returns:
        V: 1D array of internal shear at each x (kips).
    """
    _validate_gussets(gussets)
    _validate_gusset_forces(forces)
    
    x = _normalize_beam_x_coordinates(np.asarray(x, dtype=float))
    L = float(np.max(x))
    
    W = uniform_load_due2_vertical_force_and_moment(gussets, forces) # kips/in
    Rl, _ = beam_reactions_due2_chevron_effect(L, x_wp, gussets, forces) # kips
    masks, intvl = _generate_beam_x_coordinate_masks(x, x_wp, gussets)

    # Baseline shear is the left reaction everywhere
    V = np.full_like(x, Rl, dtype=float)  # kips

    for key, gusset in gussets.items():
        Lg = float(gusset.Lg)
        i = intvl[key] # intervals
        
        Wl = float(W[key]["left"])   # over [L1, x_wp], kips/in
        Wr = float(W[key]["right"])  # over [x_wp, L2], kips/in
        
        _, mB, mC, mD = masks[key]
        
        V[mB] += Wl * (x[mB] - i[0])
        V[mC] += 0.5 * Wl * Lg + Wr * (x[mC] - i[1])
        V[mD] += 0.5 * Wl * Lg + 0.5 * Wr * Lg
            
    return V


def beam_internal_moment_due_to_chevron_effect(x: np.ndarray, x_wp: float, gussets: Dict[str, GussetPlate], forces: Dict[str, Dict[str, Force]]) -> np.ndarray:
    """Computes beam internal moment induced by the forces exerted by connected chevron braces.

    Args:
        x (np.ndarray): The x-coordinates along the beam (inches).
        x_wp (float): Distance from the left beam support to the work point (inches).
        gussets (dict): A dictionary containing GussetPlate objects for 'top' and 'bottom' gussets.
        forces (dict): A dictionary containing Force objects for 'left' and 'right' braces connected to the gusset.
            {
                'top': {'left': Force, 'right': Force}, \n
                'bottom': {'left': Force, 'right': Force}
            }

    Returns:
        np.ndarray: The internal moment at each x-coordinate along the beam (kips-in).
    """
    _validate_gussets(gussets)
    _validate_gusset_forces(forces)
    
    x = _normalize_beam_x_coordinates(np.asarray(x, dtype=float))
    L = float(np.max(x))
    
    Q = uniform_moment_due2_horizontal_force(gussets, forces) # kip-in/in
    W = uniform_load_due2_vertical_force_and_moment(gussets, forces) # kips/in
    Rl, _ = beam_reactions_due2_chevron_effect(L, x_wp, gussets, forces) # kips

    # Baseline moment is the left reaction times distance
    M = np.full_like(x, Rl * x, dtype=float)  # kips-in
    
    masks, intvl = _generate_beam_x_coordinate_masks(x, x_wp, gussets)
    
    for key, gusset in gussets.items():
        Lg = float(gusset.Lg)
        i = intvl[key] # intervals

        Wl = float(W[key]["left"])   # over [L1, x_wp], kips/in
        Wr = float(W[key]["right"])  # over [x_wp, L2], kips/in
        
        _, mB, mC, mD = masks[key]
        
        M[mB] += Q[key] * (x[mB] - i[0]) + 0.5 * Wl * (x[mB] - i[0]) ** 2
        M[mC] += Q[key] * (x[mC] - i[0]) + 0.5 * Wl * Lg * (x[mC] - i[1] + Lg / 4) + 0.5 * Wr * (x[mC] - i[1]) ** 2
        M[mD] += Q[key] * Lg + 0.5 * Wl * Lg * (x[mD] - i[1] + Lg / 4) + 0.5 * Wr * Lg * (x[mD] - i[2] + Lg / 4)

        # M[mA] = Rl * x[mA]
        # M[mB] = Rl * x[mask_B] + 0.5 * Wl * (x[mask_B] - L1)**2 + Q * (x[mask_B] - L1)

        # if np.any(mask_C):
        #     M[mask_C] = Rl * x[mask_C] + 0.5 * Wl * Lg * (x[mask_C] - L1 - Lg / 4) + 0.5 * Wr * (x[mask_C] - L1 - Lg / 2) ** 2 + Q * (x[mask_C] - L1)

        # if np.any(mask_D):
        #     M[mask_D] = Rr * (L - x[mask_D])

    return M
    
    
# Utility classes and functions

class Force:
    def __init__(self, P: float, theta: float) -> None:
        """
        Initialize a Force object.

        Args:
            P (float): The magnitude of the force (kips).
            theta (float): The angle of the force from horizontal (degrees).
        """
        self.P = P
        self.theta = theta
    
    @property
    def X(self):
        "Horizontal component of the force."
        return self.P * cos(radians(self.theta))
    
    @property
    def Y(self):
        "Vertical component of the force."
        return self.P * sin(radians(self.theta))


def _validate_gussets(gussets: Dict[str, GussetPlate]) -> None:
    accepted_keys = {'top', 'bottom'}
    for key in gussets.keys():
        if not isinstance(gussets[key], GussetPlate):
            raise ValueError(f"Value for key '{key}' in gussets dictionary must be a GussetPlate object.")
        if key not in accepted_keys:
            raise ValueError(f"Invalid key '{key}' in gussets dictionary. Accepted keys are: {accepted_keys}")
        
        
def _validate_gusset_forces(forces: Dict[str, Dict[str, Force]]) -> None:
    v_keys = ('top', 'bottom')
    h_keys = ('left', 'right')
    
    for k1 in forces.keys():
        if k1 not in v_keys:
            raise ValueError(f"Invalid key '{k1}' in forces dictionary. Accepted keys are: {v_keys}")
        for k2 in forces[k1].keys():
            if k2 not in h_keys:
                raise ValueError(f"Invalid key '{k2}' in forces dictionary. Accepted keys are: {h_keys}")
            if not isinstance(forces[k1][k2], Force):
                raise ValueError(f"Value for key '{k2}' in forces dictionary must be a Force object.")
            

def _validate_beam_x_coordinates(x: np.ndarray) -> None:
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a numpy ndarray.")
    if x.ndim != 1:
        raise ValueError("x must be a 1D array of coordinates.")
    if len(x) < 2:
        raise ValueError("x must contain at least two coordinates to define a beam span.")


def _normalize_beam_x_coordinates(x: np.ndarray) -> np.ndarray:
    _validate_beam_x_coordinates(x)
    x0 = float(np.min(x))
    x_shift = x - x0
    return x_shift


def _generate_beam_x_coordinate_masks(x: np.ndarray,x_wp: float, gussets: Dict[str, "GussetPlate"]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build four boolean masks per gusset footprint:
      A: x < L1
      B: L1 <= x < x_wp
      C: x_wp <= x < L2
      D: x >= L2
    where L1 = x_wp + Δ - Lg/2, L2 = x_wp + Δ + Lg/2.

    Args:
        x (np.ndarray): 1D array of x-coordinates (inches).
        x_wp (float): Distance from beam left support to work point (w.p.) (inches).
        gussets (dict[GussetPlate]): Dictionary of gusset plates.

    Returns:
        dict[str, tuple]: Dictionary of masks for each gusset plate.
        
            {
                'top': (masks for top gusset), \n
                'bottom': (masks for bottom gusset)
            }
    """
    _normalize_beam_x_coordinates(np.asarray(x, dtype=float))
    _validate_gussets(gussets)
    
    itvl = {} # intervals
    masks = {}
    for key, gusset in gussets.items():
        Lg = float(gusset.Lg)
        Delta = float(gusset.loc[0])
        L1 = x_wp + Delta - 0.5 * Lg
        L2 = x_wp + Delta + 0.5 * Lg
        
        maskA = (x < L1)
        maskB = (x >= L1) & (x < x_wp + Delta)
        maskC = (x >= x_wp + Delta) & (x < L2)
        maskD = (x >= L2)
        
        masks[key] = (maskA, maskB, maskC, maskD)
        itvl[key] = (L1, x_wp + Delta, L2)
        
    return masks, itvl