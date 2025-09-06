from scipy.optimize import fsolve

  
    
def _moment_area_terms(length: float, x: float, pin_rigid_length: float, fix_rigid_length: float):
    """
    Compute moment areas and lever arms for a beam with pinned and fixed ends.
    
    Args:
        length (float): Total beam length.
        x (float): Distance from pinned end (candidate inflection point).
        pin_rigid_length (float): Length of rigid zone at pinned end.
        fix_rigid_length (float): Length of rigid zone at fixed end.

    Returns:
        tuple: (pin_moment_area, fix_moment_area, pin_lever_arm, fix_lever_arm)
    """
    # Moment at pinned and fixed ends
    pin_moment = 1
    fix_moment = (length - x) * pin_moment / x

    # Moment offsets at pinned and fixed ends due to rigid zones
    pin_rigid_moment = (x - pin_rigid_length) * pin_moment / x
    fix_rigid_moment = (length - x - fix_rigid_length) * fix_moment / (length - x)

    # Moment areas
    pin_moment_area = pin_rigid_moment * (x - pin_rigid_length) / 2
    fix_moment_area = fix_rigid_moment * (length - x - fix_rigid_length) / 2

    # Lever arms (distance from pinned end to centroid of moment areas)
    pin_lever_arm = pin_rigid_length + (x - pin_rigid_length) / 3
    fix_lever_arm = x + (length - x - fix_rigid_length) * 2 / 3

    return pin_moment_area, fix_moment_area, pin_lever_arm, fix_lever_arm


def get_pinned_fixed_inflection(length: float, pin_rigid_length: float, fix_rigid_length: float):
    """Return the location of inflection point from the pinned end in a beam with pinned and fixed ends. The beam is assumed prismatic with constant EI. The computation is performed using the moment-area method.

    Args:
        length (_type_): total length of the beam.
        pin_rigid_length (_type_): length of the rigid zone at the pinned end.
        fix_rigid_length (_type_): length of the rigid zone at the fixed end.
        
    Returns:
        float: location of inflection point from the pinned end.
    """
    def equation(x):
        
        pin_area, fix_area, pin_arm, fix_arm = _moment_area_terms(length, x, pin_rigid_length, fix_rigid_length)
        
        # Defletion at pinned end
        return pin_area * pin_arm - fix_area * fix_arm
    
    # Solve for x to obtain zero deflection at pinned end
    return fsolve(equation, length / 2)[0]


def get_pinned_fixed_carry_over_factor(length: float, pin_rigid_length: float, fix_rigid_length: float):
    """Return the carry-over (CO)factor for a beam with pinned and fixed ends. It shows what portion of the applied moment at the pinned end is carried over to the fixed end. The beam is assumed prismatic with constant EI.

    Args:
        length (float): total length of the beam.
        pin_rigid_length (float): length of the rigid zone at the pinned end.
        fix_rigid_length (float): length of the rigid zone at the fixed end.

    Returns:
        float: carry-over factor.
    """
    # Calculate the carry-over factor using the moment-area method
    x_inf = get_pinned_fixed_inflection(length, pin_rigid_length, fix_rigid_length)
    return (length - x_inf) / x_inf


def get_pinned_fixed_stiffness_factor(length: float, pin_rigid_length: float, fix_rigid_length: float):
    """Return the stiffness factor (K) for a beam with pinned and fixed ends.

    Args:
        length (float): total length of the beam.
        pin_rigid_length (float): length of the rigid zone at the pinned end.
        fix_rigid_length (float): length of the rigid zone at the fixed end.

    Returns:
        float: stiffness factor.
    """
    """Return the stiffness factor (K) for a beam with pinned and fixed ends."""
    x_inf = get_pinned_fixed_inflection(length, pin_rigid_length, fix_rigid_length)
    pin_area, fix_area, _, _ = _moment_area_terms(length, x_inf, pin_rigid_length, fix_rigid_length)
    
    # Stiffness factor
    return length / (pin_area - fix_area)



def get_fixed_end_moment_uniform_load(uniform_load: float, length: float, left_rigid_length: float, right_rigid_length: float):
    rigid_end_moment = uniform_load * (length - left_rigid_length - right_rigid_length) ** 2 / 12
    rigid_end_shear = uniform_load * (length - left_rigid_length - right_rigid_length) / 2
    
    left_end_shear = rigid_end_shear * (length/2) / (length/2 - left_rigid_length)
    left_end_moment = rigid_end_moment + (rigid_end_shear + left_end_shear) * left_rigid_length / 2
    
    right_end_shear = rigid_end_shear * (length/2) / (length/2 - right_rigid_length)
    right_rigid_moment = rigid_end_moment + (rigid_end_shear + right_end_shear) * right_rigid_length / 2
    
    return left_end_moment, -right_rigid_moment

# Main function =========================================================
def main():
    pass
    
    
if __name__ == "__main__":
    main()