################################
# Concrete Shear Module
# Written by: Hossein Karagah
# Date: 2025-08-22
# Description: This module provides classes and functions to compute concrete shear strength per ACI 318.
################################
import numpy as np
import assets.modules.concrete as concrete
from assets.modules.steel import LongitudinalRebar, TransverseRebar

def get_size_effect_mod_factor(d:float):
    """Computes the size effect modification factor, λs, per ACI 318-19 Section 22.5.5.1.3.

    Args:
        d (float): distance from extreme compression fiber to centroid of tension reinforcement (inches).

    Returns:
        float: size effect modification factor, λs.
    """
    return min(1.0, 2/(1 + d/10))


def get_non_prestressed_min_shear_rebar_ratio(bw:float, fc:float, fyt:float):
    """Computes the minimum shear reinforcement ratio, A_v,min/s, for non-prestressed concrete members per ACI 318-19 Section 7.6.3.3 for one-way slabs, 9.6.3.4 for beams, or 10.6.2.2 for columns.

    Args:
        bw (float): web width or circular section diameter (inches).
        fc (float): specified compressive strength of concrete (psi).
        fyt (float): specified yield strength of transverse reinforcement (psi).

    Returns:
        float: minimum shear reinforcement ratio, A_v,min/s.
    """
    return max(
        0.75 * np.sqrt(fc) * bw / fyt,
        50 * bw / fyt
    )
    
    
def get_prestressed_min_shear_rebar_ratio(d:float, bw:float, fc:float, fyt:float, Aps:float, As:float, fse:float, fpu:float, fy:float):
    """Computes the minimum shear reinforcement ratio, A_v,min/s, for prestressed concrete members per ACI 318-19 Section 9.6.3.4 for beams.
    
    Args:
        d (float): distance from extreme compression fiber to centroid of tension reinforcement (inches).
        bw (float): web width or circular section diameter (inches).
        fc (float): specified compressive strength of concrete (psi).
        fyt (float): specified yield strength of transverse reinforcement (psi).
        Aps (float): area of prestressed longitudinal tension reinforcement (in^2).
        As (float): area of non-prestressed longitudinal tension reinforcement (in^2).
        fse (float): effective stress in prestressed reinforcement after allowance for all prestress losses (psi).
        fpu (float): specified tensile strength of prestressing reinforcement (psi).
        fy (float): specified yield strength of non-prestressed reinforcement (psi).

    Returns:
        float: minimum shear reinforcement ratio, A_v,min/s.
    """
    if Aps * fse < 0.4 * (Aps * fpu + As * fy):
        return get_non_prestressed_min_shear_rebar_ratio(bw, fc, fyt)
    else:
        return min(
            get_non_prestressed_min_shear_rebar_ratio(bw, fc, fyt),
            Aps * fpu * np.sqrt(d / bw) / (80 * fyt * d)
        )
        

def get_longitudinal_rebar_ratio(beam: concrete.UniaxialBending, neutral_depth: float) -> float:
    """Computes the longitudinal reinforcement ratio, ρw, for a given beam and neutral depth.

    Args:
        beam (concrete.UniaxialBending): The concrete beam object.
        neutral_depth (float): The neutral depth of the section (inches).

    Returns:
        float: Longitudinal reinforcement ratio, ρw.
    """
    long_tensile_bars = beam.get_tensile_rebars(neutral_depth)
    As = sum(bar.area for bar in long_tensile_bars)
    bw = beam.section_width()
    d = beam.get_tensile_effective_depth(neutral_depth)
    return As / (bw * d)


def get_non_prestressed_min_shear_strength(beam: concrete.UniaxialBending, neutral_depth: float, is_lightweight: bool = False) -> float:
    """Computes the shear strength of non-prestressed concrete members per ACI 318-19 Section 22.5.5.1.1.
    
    Args:
        beam (concrete.UniaxialBending): The concrete beam object.
        neutral_depth (float): The neutral depth of the section (inches).
        is_lightweight (bool, optional): Whether the concrete is lightweight. Defaults to False.
        
    Returns:
        float: minimum shear strength of non-prestressed concrete member (lbs).
    """
    lambda_ = concrete.lightweight_factor(is_lightweight)
    fc = beam.sec.mat.fc
    bw = beam.section_width()
    d = beam.get_tensile_effective_depth(neutral_depth)
    return lambda_ * np.sqrt(fc) * bw * d


def get_non_prestressed_max_shear_strength(beam: concrete.UniaxialBending, neutral_depth: float, is_lightweight: bool=False) -> float:
    """Computes the maximum shear strength of non-prestressed concrete members per ACI 318-19 Section 22.5.5.1.1.
    
    Args:
        beam (concrete.UniaxialBending): The concrete beam object.
        neutral_depth (float): The neutral depth of the section (inches).
        is_lightweight (bool, optional): Whether the concrete is lightweight. Defaults to False.
        
    Returns:
        float: maximum shear strength of non-prestressed concrete member (lbs).
    """
    return 5 * get_non_prestressed_min_shear_strength(beam, 0, neutral_depth, not is_lightweight)


def get_non_prestressed_shear_strength(beam: concrete.UniaxialBending, axial_force: float, neutral_depth: float, is_lightweight: bool = True) -> float:
    """Computes the shear strength of non-prestressed concrete members per ACI 318-19 Section 22.5.5.
    """
    pass