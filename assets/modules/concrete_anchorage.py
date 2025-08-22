################################
# Concrete Section Module
# Written by: Hossein Karagah
# Date: 2025-07-11
# Description: This module provides classes and functions to compute concrete anchorage parameters per ACI 318. The units are in US Customary.
################################


import os
import sys
sys.path.append(os.path.abspath("../../.."))
from assets.modules.materials import Concrete, ACIConcrete, BilinearSteel
from assets.modules.steel import LongitudinalRebar, TransverseRebar
from assets.modules.shapes import *
import assets.modules.concrete as concrete




class Anchor:
    def __init__(self, type: str):
        self.type = self._validate_type(type) # "cast-in" or "post-installed"

    def _validate_type(self, type: str) -> str:
        if type.lower() not in ["cast-in", "post-installed"]:
            raise ValueError("Anchor type must be either 'cast-in' or 'post-installed'.")
        return type.lower()
    




def lambda_a(mat: Concrete, failure_mode: str) -> float:
    """Returns modification factor to reflect the reduced mechanical properties of lightweight concrete in certain concrete anchorage applications per ACI 318-25 §17.2.4.
    
    Args:
        mat (Concrete): The concrete material object.
        failure_mode (str): The failure mode, can be 'concrete', 'anchor', or 'adhesive'.
    """
    if failure_mode.lower() == 'concrete':
        return 1.0 * mat.lambda_
    elif failure_mode.lower() == 'anchor':
        return 0.8 * mat.lambda_
    elif failure_mode.lower() == 'adhesive':
        return 0.6 * mat.lambda_
    else:
        raise ValueError("Invalid failure mode. Choose 'concrete', 'anchor', or 'adhesive'.")
    

def psi_a_tension(anchor: Anchor, supp_rebar: bool):
    """Returns anchor strength modification factor in tension based on anchor type and supplementary reinforcement per ACI 318-25 §17.5.4.1.
    
    Args:
        anchor (Anchor): The anchor object
        supp_rebar (bool): Whether supplementary reinforcement is present
    """
    values = {
        "cast-in": {
            True: 1.0,  # with supplementary reinforcement
            False: 0.95 # without supplementary reinforcement
        },
        "post-installed": {
            True: {"cat-1": 1.0, "cat-2": 0.85, "cat-3": 0.75},  # with supplementary reinforcement
            False: {"cat-1": 0.85, "cat-2": 0.75, "cat-3": 0.60}  # without supplementary reinforcement
        }
    }
    
    return values[anchor.type][supp_rebar]    
    
def psi_a_shear(anchor: Anchor, supp_rebar: bool):
    return 1.0 if supp_rebar else 0.95
    
    
def psi_a(anchor: Anchor, applied_force: str, supp_rebar: bool):
    if applied_force.lower() not in ["tension", "shear"]:
        raise ValueError("Applied force must be either 'tension' or 'shear'.")

    if applied_force.lower() == "tension":
        return psi_a_tension(anchor, supp_rebar)
    else:  # applied_force.lower() == "shear"
        return psi_a_shear(anchor, supp_rebar)
        
