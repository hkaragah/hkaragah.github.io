from typing import Optional, List, Union
import uuid
from scipy.optimize import fsolve
from dataclasses import dataclass, field
from assets.modules.shapes import Rectangle, Circle, Point, TSection
from assets.modules.materials import Concrete, ACIConcrete, Steel
import numpy as np
  
def _generate_id(prefix: str) -> str:
    """Generate a short unique identifier with a given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

class IDGenerator:
    counters = {'C':0, 'B':0, 'J':0}

    @classmethod
    def generate(cls, prefix: str) -> str:
        cls.counters[prefix] += 1
        return f"{prefix}{cls.counters[prefix]}"


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
        length (float): total length of the beam.
        pin_rigid_length (float): length of the rigid zone at the pinned end.
        fix_rigid_length (float): length of the rigid zone at the fixed end.
        
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


def get_joint_distribution_factors(kFactors: list):
    """Convert stiffness factors to distribution factors. The sum of the distribution factors is 1. The distribution factor for each member is its stiffness factor divided by the total stiffness of all members connected to the joint.
    
    Args:
        kFactors (list): list of stiffness factors for each member connected to the joint.

    Returns:
        list: distribution factors for each member connected to the joint.
    """
    total = sum(kFactors)
    return [k / total for k in kFactors]

@dataclass
class ConcreteColumn:
    length: float
    width: float
    depth: float # out-of-plane dimension
    mat: ACIConcrete
    joint: Optional['Concrete2DJoint'] = None # Back-link to joint
    id: str = field(init=False)
    sec: Rectangle = field(init=False)
    
    
    def __post_init__(self):
        self.id = IDGenerator.generate("C")
        self.sec = Rectangle(width=self.width, height=self.depth)
    
    @property
    def rigid_length(self):
        """Forwarded rigid length from the joint."""
        return self.joint.column_rigid_length if self.joint else None
    
    def __repr__(self):
        return f"Column {self.id} @ {self.joint.id if self.joint else None} [height={self.length:.2f}', {self.width*12}\" x {self.depth*12}\"]"
    
    def __str__(self):
        return f"Column {self.id} @ {self.joint.id if self.joint else None} [height={self.length:.2f}', {self.width*12}\" x {self.depth*12}\"]"

@dataclass
class ConcreteRectangleBeam:
    length: float
    width: float # out-of-plane dimension
    depth: float
    mat: ACIConcrete
    start_joint: Optional['Concrete2DJoint'] = None
    end_joint: Optional['Concrete2DJoint'] = None
    id: str = field(init=False)
    sec: Rectangle = field(init=False)
    
    def __post_init__(self):
        self.id = IDGenerator.generate("B")
        self.sec = Rectangle(width=self.width, height=self.depth)

    @property
    def start_rigid_length(self):
        """Rigid length at the start joint (0 if no start joint)."""
        return self.start_joint.beam_rigid_length if self.start_joint else 0

    @property
    def end_rigid_length(self):
        """Rigid length at the end joint (0 if no end joint)."""
        return self.end_joint.beam_rigid_length if self.end_joint else 0
    
    def __repr__(self):
        return f"Beam {self.id} @ {self.start_joint.id if self.start_joint else None} --> {self.end_joint.id if self.end_joint else None} [length={self.length:.2f}', {self.width*12}\" x {self.depth*12}\"]"

    def __str__(self):
        return f"Beam {self.id} @ {self.start_joint.id if self.start_joint else None} --> {self.end_joint.id if self.end_joint else None} [length={self.length:.2f}', {self.width*12}\" x {self.depth*12}\"]"

@dataclass
class ConcreteTSectionBeam:
    length: float
    width: float
    depth: float
    flange_width: float
    flange_thickness: float
    mat: Concrete
    start_joint: Optional['Concrete2DJoint'] = None
    end_joint: Optional['Concrete2DJoint'] = None
    id: str = field(init=False)
    sec: TSection = field(init=False)

    def __post_init__(self):
        self.id = IDGenerator.generate("B")
        self.sec = TSection(total_height=self.depth, web_width=self.width, flange_width=self.flange_width, flange_thickness=self.flange_thickness)
        
    @property
    def start_rigid_length(self):
        """Rigid length at the start joint (0 if no start joint)."""
        return self.start_joint.beam_rigid_length if self.start_joint else 0
    
    @property
    def end_rigid_length(self):
        """Rigid length at the end joint (0 if no end joint)."""
        return self.end_joint.beam_rigid_length if self.end_joint else 0
    
    def __repr__(self):
        return f"T-Beam {self.id} @ {self.start_joint.id if self.start_joint else None} --> {self.end_joint.id if self.end_joint else None} [length={self.length:.2f}', web: {self.width*12}\" x {self.depth*12}\", flange: {self.flange_width*12}\" x {self.flange_thickness*12}\"]"
    
    def __str__(self):
        return f"T-Beam {self.id} @ {self.start_joint.id if self.start_joint else None} --> {self.end_joint.id if self.end_joint else None} [length={self.length:.2f}', web: {self.width*12}\" x {self.depth*12}\", flange: {self.flange_width*12}\" x {self.flange_thickness*12}\"]"

@dataclass
class Concrete2DJoint:
    left_beam: Optional[Union[ConcreteRectangleBeam, ConcreteTSectionBeam]] = None
    right_beam: Optional[Union[ConcreteRectangleBeam, ConcreteTSectionBeam]] = None
    lower_column: Optional[ConcreteColumn] = None
    upper_column: Optional[ConcreteColumn] = None
    id: str = field(init=False)

    def __post_init__(self):
        self.id = IDGenerator.generate("J")
        # Collect beams and columns in internal lists
        self._beams: List["ConcreteRectangleBeam"] = [b for b in [self.left_beam, self.right_beam] if b]
        self._columns: List["ConcreteColumn"] = [c for c in [self.upper_column, self.lower_column] if c]
        self._elements: List = self._beams + self._columns

        # Validation
        if len(self._beams) < 1 or len(self._beams) > 2:
            raise ValueError("A 2D joint must connect to 1 or 2 beams.")
        if len(self._columns) > 2:
            raise ValueError("A joint can connect to a maximum of 2 columns.")
        
        # Back-link beams
        for b in self._beams:
            if b.start_joint is None:
                b.start_joint = self
            elif b.end_joint is None:
                b.end_joint = self
            else:
                raise ValueError(f"Beam {b.id} is already connected to two joints.")

        # Back-link columns
        for c in self._columns:
            if c.joint is not None:
                raise ValueError(f"Column {c.id} is already linked to a joint.")
            c.joint = self
            
    def __repr__(self):
        return f"2D Joint {self.id} @ {self.left_beam.id if self.left_beam else None}← →{self.right_beam.id if self.right_beam else None}, {self.lower_column.id if self.lower_column else None}↓ ↑{self.upper_column.id if self.upper_column else None}"
    
    def __str__(self):
        return f"2D Joint {self.id} @ {self.left_beam.id if self.left_beam else None}← →{self.right_beam.id if self.right_beam else None}, {self.lower_column.id if self.lower_column else None}↓ ↑{self.upper_column.id if self.upper_column else None}"

    @property
    def beam_rigid_length(self):
        """Return half of the sum of connected column widths, or 0 if no columns."""
        if not self._columns:
            return 0
        return sum([c.width / 2 for c in self._columns]) / len(self._columns)
        
    @property
    def column_rigid_length(self):
        """Return half of the sum of connected beam heights, or None if no beams."""
        if not self._beams:
            return None
        return sum([b.depth / 2 for b in self._beams]) / len(self._beams)
    
    @property
    def stiffness_factors(self):
        """Return stiffness factors for connected beams."""
        k_factors = {}
        for b in self._beams:
            EI = b.mat.E * (144/1e3) * b.sec.moment_inertia()[0]
            k = get_pinned_fixed_stiffness_factor(b.length, b.start_rigid_length, b.end_rigid_length)
            k_factors[b.id] = k * EI / b.length
        for c in self._columns:
            EI = c.mat.E * (144/1e3) * c.sec.moment_inertia()[0]
            k = get_pinned_fixed_stiffness_factor(c.length, c.rigid_length, 0)
            k_factors[c.id] = k * EI / c.length
        return k_factors
    
    @property
    def distribution_factors(self):
        """Return distribution factors for connected beams and columns."""
        k_factors = list(self.stiffness_factors.values())
        d_factors = get_joint_distribution_factors(k_factors)
        return dict(zip(self.stiffness_factors.keys(), d_factors))
    
    @property
    def carry_over_factors(self):
        """Return carry-over factors for connected beams."""
        co_factors = {}
        for b in self._beams:
            co = get_pinned_fixed_carry_over_factor(b.length, b.start_rigid_length, b.end_rigid_length)
            co_factors[b.id] = co
        for c in self._columns:
            co = get_pinned_fixed_carry_over_factor(c.length, c.rigid_length, 0)
            co_factors[c.id] = co
        return co_factors
    

def get_beam_reactions(length: float, uniform_load: float, left_moment: float, right_moment: float):
    """Return the reactions at the left and right supports of a beam with a uniform load and end moments. Ensure units are consistent (e.g., ft for length, kip/ft for uniform load, and kip-ft for moments).

    Args:
        length (float): length of the beam.
        uniform_load (float): uniform load applied to the beam. Positive is downward.
        left_moment (float): moment applied at the left support. Positive is ccw.
        right_moment (float): moment applied at the right support. Positive is ccw.

    Returns:
        tuple: reactions at the left and right supports.
    """
    left_shear = (uniform_load * length / 2) + (left_moment / length) + (right_moment / length)
    right_shear = (uniform_load * length / 2) - (left_moment / length) - (right_moment / length)
    return left_shear, -right_shear


def generate_linear_shear_diagram(
    beam: Union[ConcreteRectangleBeam, ConcreteTSectionBeam],
    uniform_load: float,
    left_shear: float,
    right_shear: float,
    n_points: int = 100,
    atol: float = 1e-12,
):
    """
    Builds (x, V) for a beam with rigid offsets and a uniform load.
    Ensures an (x, V) pair is included where V == 0 (if it lies in the active span).
    
    Args:
        beam (ConcreteRectangleBeam or ConcreteTSectionBeam): The beam object.
        uniform_load (float): Magnitude of the uniform load applied to the beam.
        left_shear (float): Shear force at the left support.
        right_shear (float): Shear force at the right support.
        n_points (int, optional): Number of points to generate along the beam. Defaults to 100.
        atol (float, optional): Absolute tolerance for zero checks. Defaults to 1e-12.
        
    Returns:
        tuple: (x, V) where x is the array of positions along the beam and V is the array of shear forces at those positions.
    """
    length = beam.length
    
    # Critical section for moment design (face of column)
    a = float(beam.start_rigid_length) 
    b = float(length - beam.end_rigid_length)
       
    if a < 0 or (length - b) < 0:
        raise ValueError("Rigid lengths must be non-negative.")
    if a > b:
        raise ValueError("Sum of rigid lengths cannot exceed the beam length.")    
    
    # Critical section for shear design per ACI 315-25, section 9.4.3.2
    a1 = float(beam.start_rigid_length + beam.depth / 2)
    b1 = float(length - beam.end_rigid_length - beam.depth / 2)
    
    n_pts = max(int(n_points), 7)  # minimum 7 points
    n_mid = n_pts - 4 # only generating points between a1 and b1

    # Build a base grid with fix size: 0 | [a, a1..b1, b] | L
    x_mid = np.linspace(a1, b1, n_mid, endpoint=True)
    x = np.concatenate(([0.0, a], x_mid, [b, length]))

    # If there is a zero crossing, move the nearest interior sample to x0 (no insertion)
    if (left_shear * right_shear) < 0 and length > atol:
        Vl, Vr = float(left_shear), float(right_shear)
        m = (right_shear - left_shear) / length  # linear slope across active span
        if abs(m) > atol:
            x0 = -left_shear / m
            if a1 <= x0 <= b1:  # only if within active span
                # choose nearest interior index (avoid endpoints at 0 and L)
                k = 1 + np.argmin(np.abs(x[1:-1] - x0))
                x[k] = x0  # replace, do not insert → length stays n_pts

    # Compute V with masks
    V = np.empty_like(x, dtype=float)
    m1 = x < a
    m2 = (x >= a) & (x <= b)
    m3 = x > b

    V[m1] = left_shear
    V[m2] = left_shear - uniform_load * x[m2]
    V[m3] = right_shear

    # Snap exact zero at (approximate) crossing sample—optional but tidy
    if (left_shear * right_shear) < 0 and (b - a) > atol:
        i0 = 1 + np.argmin(np.abs(x[1:-1] - np.clip(a - float(left_shear) / ((right_shear - left_shear)/(b - a)), a, b)))
        V[i0] = 0.0

    return x, V


def generate_quadratic_moment_diagram(x: np.ndarray, shear: np.ndarray, left_moment: float, right_moment: float):
    """Generate moment diagram for a beam given shear diagram and end moments.
    Args:
        x (np.ndarray): Array of positions along the beam.
        shear (np.ndarray): Array of shear forces at the positions in x.
        left_moment (float): Moment at the left support.
        right_moment (float): Moment at the right support.
        
    Returns:
        tuple: (x, M) where x is the array of positions along the beam and M is the array of moments at those positions.
    """
    moment = np.zeros_like(x)
    moment[0] = -left_moment

    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        avg_shear = (shear[i] + shear[i-1]) / 2
        moment[i] = moment[i-1] + avg_shear * dx

    moment[-1] = right_moment

    return x, moment    


def generate_shear_moment_diagrams(beam: Union[ConcreteRectangleBeam, ConcreteTSectionBeam], uniform_load: float, support_moments:tuple[float, float], n_points: int = 100, atol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate shear and moment diagrams for a beam with rigid offsets, uniform load, and end moments.
    
    Args:
        beam (ConcreteRectangleBeam or ConcreteTSectionBeam): The beam object.
        uniform_load (float): Magnitude of the uniform load applied to the beam.
        support_moments (tuple): (left_moment, right_moment) applied at the supports. Positive is ccw.
        n_points (int, optional): Number of points to generate along the beam. Defaults to 100.
        atol (float, optional): Absolute tolerance for zero checks. Defaults to 1e-12.
        
    Returns:
        tuple: (x, V, M) where x is the array of positions along the beam, V is the array of shear forces, and M is the array of moments at those positions.
    """
    left_moment, right_moment = support_moments
    
    left_shear, right_shear = get_beam_reactions(beam.length, uniform_load, left_moment, right_moment)
    x, V = generate_linear_shear_diagram(beam, uniform_load, left_shear, right_shear, n_points=n_points, atol=atol)
    x, M = generate_quadratic_moment_diagram(x, V, left_moment, right_moment)
    return x, V, M


def generate_moment_distribution_input(ordered_joints: list[Concrete2DJoint], uniform_loads: dict[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    DF = np.zeros((len(ordered_joints), 2), dtype=float)
    COF = np.zeros((len(ordered_joints), 2), dtype=float)
    FEM = np.zeros((len(ordered_joints), 2), dtype=float)
    
    for i, joint in enumerate(ordered_joints):
        for j, member in enumerate(joint._beams):
            DF[i, j] = joint.distribution_factors.get(member.id, 0)
            COF[i, j] = joint.carry_over_factors.get(member.id, 0)
            w = uniform_loads.get(member.id, 0)
            left_fem, right_fem = get_fixed_end_moment_uniform_load(w, member.length, member.start_rigid_length, member.end_rigid_length)
            if member.start_joint == joint:
                FEM[i, j] = left_fem
            if member.end_joint == joint:
                FEM[i, j] = right_fem
        
    return DF, COF, FEM


def distribute_moments(ordered_joints: list[Concrete2DJoint], uniform_loads: dict[float], tol: float = 1e-3, max_iter: int = 100):
    
    DF, COF, FEM = generate_moment_distribution_input(ordered_joints, uniform_loads)

    total_moments = FEM.copy()
    
    for _ in range(max_iter):
        # Calculate moment distribution
        unbalanced_moments = total_moments.sum(axis=1, keepdims=True)
        distributed_moments = DF * -unbalanced_moments
        carried_moments = COF * distributed_moments
        # in-place swap of the first two columns
        carried_moments[:, [0, 1]] = carried_moments[:, [1, 0]]
        total_moments += unbalanced_moments + carried_moments
        
    return total_moments


# Main function =========================================================
def main():
    pass
    
    
if __name__ == "__main__":
    main()