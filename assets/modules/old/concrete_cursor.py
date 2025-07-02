################################
# Concrete Section Module
# Written by: Hossein Karagah
# Date: 2025-03-10
# Description: This module provides classes and functions to model and analyze rectangular concrete sections with rebars.
################################

import numpy as np
from copy import deepcopy
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Union, Optional, Callable
from scipy.optimize import root_scalar, fsolve

from assets.modules.shapes import Point, Rectangle, Circle, get_circle_group_centroid
from assets.modules.materials import ACIConcrete, Concrete
from assets.modules.steel import LongitudinalRebar, TransverseRebar


# Constants and Enums ##########################################

# Magic numbers and string literals used throughout the module
DEFAULT_SECTION_CENTER = (0.0, 0.0)
MIN_COVER_RATIO = 0.0  # example, adjust as needed

@dataclass
class SolverConfig:
    """Configuration for the equilibrium solver."""
    method: str = 'secant'
    tolerance: float = 1e-2
    max_iterations: int = 100
    n_discretization_points: int = 100

class BendingDirection(Enum):
    """Enumeration for bending directions."""
    POSITIVE_X = "x"
    NEGATIVE_X = "-x"
    POSITIVE_Y = "y"
    NEGATIVE_Y = "-y"

    def get_primary_axis(self) -> str:
        """Returns the primary axis for the bending direction."""
        return 'y' if self in [self.POSITIVE_X, self.NEGATIVE_X] else 'x'


# Main Classes ################################################

class RectangleSection:
    """Base class for a rectangular concrete section."""
    def __init__(self, shape: Rectangle, mat: Concrete, cover: float, center: Union[Point, Tuple[float, float]] = (0.0, 0.0)):
        self._shape = deepcopy(shape)
        self._shape.center = center
        self._mat = mat
        if shape.width - 2 * cover <= 0 or shape.height - 2 * cover <= 0:
            raise ValueError("Cover is too large relative to the section dimensions.")
        self._cover = cover

    @property
    def mat(self) -> Concrete:
        """Returns the concrete material of the section."""
        return self._mat

    @property
    def shape(self) -> Rectangle:
        """Returns the rectangle shape of the section."""
        return self._shape

    @property
    def cover(self) -> float:
        """Returns the cover of the concrete section."""
        return self._cover

    @property
    def gross_area(self) -> float:
        """Returns the gross area of the section."""
        return self._shape.area


class RectangleReinforcedSection(RectangleSection):
    """A class representing a rectangular concrete beam section.
    Inherits from RectangleSection and is used to model concrete beams with rebars.
    """
    
    def __init__(self, shape: Rectangle, mat: Concrete, cover: float, center: Union[Point, Tuple[float, float]] = (0.0, 0.0)):
        super().__init__(shape, mat, cover, center)
        self._lBars: List[LongitudinalRebar] = []
        self._tBars: List[TransverseRebar] = []   
        
    @property
    def lBars(self) -> List[LongitudinalRebar]:
        """Returns the list of longitudinal rebars in the beam section."""
        return self._lBars
    
    @property
    def tBars(self) -> List[TransverseRebar]:
        """Returns the list of transverse rebars (stirrups) in the beam section."""
        return self._tBars

    def add_tBar(self, tbar: TransverseRebar):
        """Adds a transverse rebar (stirrup) to the concrete section.
        The transverse rebar is added to the list of transverse bars in the section.
        A rectangular section can have multiple transverse rebars, and each can be looped or single legged.

        Args:
            bar (TransverseRebar): Transverse rebar object to be added.
            
        """
        if not isinstance(tbar, TransverseRebar):
            raise TypeError("tBar must be an instance of TransverseRebar.")
        self._tBars.append(deepcopy(tbar))
        
    def add_lBar(self, lbar: LongitudinalRebar, center: Union[Point, Tuple[float, float]]):
        """Adds a rebar to the wall.
        The rebar is added at the specified center point. The function checks if the new rebar overlaps with existing rebars and if it is within the wall bounds minus cover.

        Args:
            lBar (LongitudinalRebar): Longitudinal reinforcement object to be added.
            center (Union[Point, Tuple[float, float]]): Center point of the longitudinal rebar.
                If a tuple is provided, it will be converted to a Point object.

        Raises:
            ValueError: If the new rebar overlaps with existing rebars or if it is outside the wall bounds minus cover.
            ValueError: If the rebar is outside the wall bounds minus cover.
        """
        if isinstance(center, tuple):
            center = Point(center[0], center[1])
        
        bar = deepcopy(lbar)
        bar.center = center
        
        # Check if new rebar overlaps with existing rebars
        for existing_rebar in self.lBars:
            if existing_rebar.shape.overlapped_with(bar.shape):
                raise ValueError(f"Rebar at {bar.center} overlaps with existing rebar at {existing_rebar.center}.")
        
        # Check if new rebar is within core region
        if not bar.shape.within(self.core()):
            x_min, x_max, y_min, y_max = self._get_bounds(self.core())
            raise ValueError(f"Rebar at {bar.center} is outside the core region x-range:({x_min}, {x_max}), y-range: ({y_min}, {y_max}).")
            
        self._lBars.append(bar)

    def add_lBar_row(self, lBar: LongitudinalRebar, center: Union[Point, Tuple[float, float]], 
                    n: int, direction: Tuple[float, float], spacing: Optional[float] = None):
        """Adds a row of rebars to the wall section.
        
        Args:
            lBar (LongitudinalRebar): Longitudinal reinforcement object to be added.
            center (Union[Point, Tuple[float, float]]): Center point of the row.
            n (int): Number of longitudinal bars in the row.
            direction (Tuple[float, float]): A 2D unit vector (x, y) specifying direction of the row.
            spacing (float): Spacing between rebars.
            
        Raises:
            ValueError: If rebars would overlap with cover region or transverse bars.
        """
        if n <= 0:
            raise ValueError("Number of bars must be positive.")
        
        # Convert tuple center to Point if needed
        center = Point(*center) if isinstance(center, tuple) else center
        
        # Normalize direction vector
        direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Direction vector cannot be zero.")
        dx, dy = direction / norm
        
        # Calculate spacing if not provided
        if spacing is None:
            spacing = self._calculate_auto_spacing(center, dx, dy, n, lBar)
        
        if spacing <= 0:
            raise ValueError("Invalid spacing: too small or negative.")
        
        # Generate and add rebars
        positions = self._generate_rebar_positions(center, dx, dy, n, spacing)
        for pos in positions:
            self.add_lBar(lBar, pos)

    def _calculate_auto_spacing(self, center: Point, dx: float, dy: float, n: int, lBar: LongitudinalRebar) -> float:
        """Calculate automatic spacing based on available space in core region."""
        x_min, x_max, y_min, y_max = self._get_bounds(self.core(lBar))
        
        # Calculate maximum distance in each direction
        max_dist_x = min(center.x - x_min, x_max - center.x) / abs(dx) if dx != 0 else float('inf')
        max_dist_y = min(center.y - y_min, y_max - center.y) / abs(dy) if dy != 0 else float('inf')
        
        max_dist = min(max_dist_x, max_dist_y)

        return max_dist / ((n - 1) / 2)

    def _get_bounds(self, core_rect: Rectangle) -> Tuple[float, float, float, float]:
        """Extract min/max bounds from core rectangle."""
        corners = core_rect.corners
        x_vals = [pt.x for pt in corners]
        y_vals = [pt.y for pt in corners]
        return min(x_vals), max(x_vals), min(y_vals), max(y_vals)

    def _generate_rebar_positions(self, center: Point, dx: float, dy: float, n: int, spacing: float) -> List[Point]:
        """Generate positions for rebars in a symmetric row."""
        positions = []
        
        if n % 2 == 1:  # Odd number: place center bar first
            positions.append(Point(center.x, center.y))
            
        pairs_to_add = n // 2

        # Add symmetric pairs
        for i in range(1, pairs_to_add + 1):
            if n % 2 == 0:
                offset = (2 * i - 1) * spacing / 2  # 0.5, 1.5, 2.5, ... * spacing
            else:
                offset = i * spacing  # 1, 2, 3, ... * spacing
            
            positions.extend([
                Point(center.x + dx * offset, center.y + dy * offset),
                Point(center.x - dx * offset, center.y - dy * offset)
            ])

        return positions
            
    def translate(self, translation: Tuple[float, float]):
        """Translates the section by dx and dy.
        The translation is applied to the shape of the section and all rebars in the section.
        The translation is specified as a tuple (dx, dy) where dx is the translation in the x direction and dy is the translation in the y direction.
        
        Args:
            translation (Tuple[float, float]): A tuple (dx, dy) specifying the translation in x and y directions.
        Raises:
            TypeError: If translation is not a tuple of two floats.
        """
        self._shape.translate(translation)
        for bar in self.lBars:
            bar._shape.translate(translation)
    
    def rotate(self, angle_deg: float, pivot: Union[Point, Tuple[float, float]] = None):
        """Rotates the section by angle_deg around the pivot point."""
        self._shape.rotate(angle_deg, pivot)
        for bar in self.lBars:
            bar._shape.rotate(angle_deg, pivot=self._shape.center) 
            
    @property
    def lBars_x_max(self):
        """Returns the maximum x coordinate of all rebars in the section."""
        if not self.lBars:
            return None
        return max([rebar.center.x for rebar in self.lBars])
    
    @property
    def lBars_x_min(self):
        """Returns the minimum x coordinate of all rebars in the section."""
        if not self.lBars:
            return None
        return min([rebar.center.x for rebar in self.lBars])
    
    @property
    def lBars_y_max(self):
        """Returns the maximum y coordinate of all rebars in the section."""
        if not self.lBars:
            return None
        return max([rebar.center.y for rebar in self.lBars])
    
    @property
    def lBars_y_min(self):
        """Returns the minimum y coordinate of all rebars in the section."""
        if not self.lBars:
            return None
        return min([rebar.center.y for rebar in self.lBars])
            

        
    @property
    def tBar_max_dia(self) -> float:
        """Returns the maximum diameter of transverse bars in the section if looped is True.
        It does not include single-legged transverse bars since they do not enclose the section.
        """
        if not self.tBars:
            return 0.0
        else:
            return max(bar.dia if bar.is_looped else 0 for bar in self.tBars)
        
    def core(self, rebar: Optional[LongitudinalRebar] = None) -> Rectangle:
        """Computes the concrete core excluding the cover, transverse bars, and optionally a rebar.
        The core is the area where the main rebars can be placed without overlapping with the cover or transverse bars.

        Args:
            rebar (Optional[RebarSection], optional): longitudinal rabar. Defaults to None.

        Returns:
            sh.Rectangle: The core rectangle shape that defines the area where longitudinal rebars can be placed.
        """
        offset = 0 if rebar is None else rebar.dia            
        width = self.shape.width - 2 * self.cover - 2 * self.tBar_max_dia - offset
        height = self.shape.height - 2 * self.cover - 2 * self.tBar_max_dia - offset
        
        return Rectangle(width, height, self._shape.center, self._shape.rotation) 
    
    @property
    def net_area(self) -> float:
        """Returns the net area of the concrete section excluding the area occupied by longitudinal rebars."""
        lBar_area = sum([bar.area for bar in self.lBars])
        return self.gross_area - lBar_area 


class GeometryAnalyzer:
    """Handles geometric calculations for concrete sections."""
    
    def __init__(self, section: RectangleReinforcedSection, direction: BendingDirection):
        self._section = section
        self._direction = direction
        
    @property
    def section(self) -> RectangleReinforcedSection:
        """Returns the concrete section."""
        return self._section
    
    @property
    def direction(self) -> BendingDirection:
        """Returns the bending direction."""
        return self._direction
    
    @property
    def section_height(self) -> float:
        """Returns the height of the section based on the bending direction."""
        if self._direction in [BendingDirection.POSITIVE_X, BendingDirection.NEGATIVE_X]:
            return self.section.shape.y_max - self.section.shape.y_min
        else:
            return self.section.shape.x_max - self.section.shape.x_min
        
    @property
    def section_width(self) -> float:
        """Returns the width of the section based on the bending direction."""
        if self._direction in [BendingDirection.POSITIVE_X, BendingDirection.NEGATIVE_X]:
            return self.section.shape.x_max - self.section.shape.x_min
        else:
            return self.section.shape.y_max - self.section.shape.y_min
    
    def get_extreme_fiber_coord(self) -> Point:
        """Returns the coordinates of the extreme compressive fiber."""
        ref_points = {
            BendingDirection.POSITIVE_X: Point(self.section.shape.center.x, self.section.shape.y_min),
            BendingDirection.NEGATIVE_X: Point(self.section.shape.center.x, self.section.shape.y_max),
            BendingDirection.POSITIVE_Y: Point(self.section.shape.x_max, self.section.shape.center.y),
            BendingDirection.NEGATIVE_Y: Point(self.section.shape.x_min, self.section.shape.center.y)
        }
        return ref_points[self._direction]
    
    def get_extreme_bar_depth(self) -> float:
        """Returns the depth of the extreme outer most rebars."""
        if not self.section.lBars:
            return None
        
        min_depth = {
            BendingDirection.POSITIVE_X: max([bar.center.y for bar in self.section.lBars]),
            BendingDirection.NEGATIVE_X: min([bar.center.y for bar in self.section.lBars]),
            BendingDirection.POSITIVE_Y: min([bar.center.x for bar in self.section.lBars]),
            BendingDirection.NEGATIVE_Y: max([bar.center.x for bar in self.section.lBars])
        }
        return min_depth[self._direction]
    
    def get_extreme_bars(self) -> List[LongitudinalRebar]:
        """Returns the list of extreme outer most tensile rebars."""
        if not self.section.lBars:
            return []
            
        depth = self.get_extreme_bar_depth()
        
        if self._direction in [BendingDirection.POSITIVE_X, BendingDirection.NEGATIVE_X]:
            return [bar for bar in self.section.lBars if bar.center.y == depth]
        else:
            return [bar for bar in self.section.lBars if bar.center.x == depth]
    
    def get_bar_depth(self) -> np.ndarray:
        """Returns the depth of each rebar from the extreme compressive fiber."""
        if not self.section.lBars:
            return np.array([])
        
        if self._direction in [BendingDirection.POSITIVE_X, BendingDirection.NEGATIVE_X]:
            return np.array([bar.center.y for bar in self.section.lBars]).reshape(-1, 1)
        else:
            return np.array([bar.center.x for bar in self.section.lBars]).reshape(-1, 1)

class StrainAnalyzer:
    """Handles strain and stress calculations for concrete sections."""
    
    def __init__(self, geometry: GeometryAnalyzer):
        self._geometry = geometry
        
    @property
    def geometry(self) -> GeometryAnalyzer:
        """Returns the geometry analyzer."""
        return self._geometry
    
    def get_curvature(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the curvature of the section."""
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        try:
            return np.divide(self.geometry.section.mat.eps_u, neutral_depth)
        except ZeroDivisionError:
            raise ValueError("Neutral depth cannot be zero.")
    
    def get_rebar_strain(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
        """Computes the strain in the longitudinal rebars."""
        if not self.geometry.section.lBars:
            return np.array([])
        
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        # Compute curvature based on neutral depth
        curv = self.get_curvature(neutral_depth)
        
        # Compute total depth of the section
        sec_height = self.geometry.section_height
        
        # Compute the depth of each rebar
        bar_depth = self.geometry.get_bar_depth()
        
        # Calculate strain for each rebar
        return curv.T * (bar_depth - (sec_height / 2 - neutral_depth.T))
    
    def get_concrete_strain(self, neutral_depth: Union[float, np.ndarray], n_points: int = 100) -> np.ndarray:
        """Computes the strain in the concrete section."""
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        # Compute curvature based on neutral depth
        curv = self.get_curvature(neutral_depth)
        
        # Compute total depth of the section
        height = self.geometry.section_height

        # Compute the depth of the center of each strip
        depth = np.linspace(-height/2 + height/(2*n_points), height/2 - height/(2*n_points), n_points).reshape(-1, 1)
        
        return curv.T * (depth - (height / 2 - neutral_depth.T))
    
    def get_rebar_stress(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
        """Computes the stress in the longitudinal rebars."""
        if not self.geometry.section.lBars:
            return np.array([])
        
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
            
        # Compute the strain in each rebar
        strain = self.get_rebar_strain(neutral_depth)
        
        # Vectorized stress calculation
        return np.array([bar.mat.stress(eps) for bar, eps in zip(self.geometry.section.lBars, strain)])
    
    def get_concrete_stress(self, neutral_depth: Union[float, np.ndarray], n_points: int = 100) -> np.ndarray:
        """Computes the stress in the concrete section."""
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        # Compute strain at the center of each strip
        strain = self.get_concrete_strain(neutral_depth, n_points)
        
        return self.geometry.section.mat.stress(strain)

class ForceAnalyzer:
    """Handles force and moment calculations for concrete sections."""
    
    def __init__(self, strain: StrainAnalyzer):
        self._strain = strain
        
    @property
    def strain(self) -> StrainAnalyzer:
        """Returns the strain analyzer."""
        return self._strain
    
    def get_rebar_force(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
        """Computes the force in the longitudinal rebars."""
        if not self.strain.geometry.section.lBars:
            return np.array([])
        
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
            
        # Compute the stress in each rebar
        stress = self.strain.get_rebar_stress(neutral_depth)
        
        # Vectorized force calculation
        areas = np.array([bar.area for bar in self.strain.geometry.section.lBars])
        return areas * stress
    
    def get_concrete_force(self, neutral_depth: Union[float, np.ndarray], n_points: int = 100) -> np.ndarray:
        """Computes the force in the concrete section."""
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        # Compute the net area of each concrete strip
        net_areas = self.get_concrete_strip_net_area(n_points)
        
        # Calculate stress at the center of each strip
        stress = self.strain.get_concrete_stress(neutral_depth, n_points)
        
        return stress * net_areas
    
    def get_concrete_strip_net_area(self, n_points: int = 100) -> np.ndarray:
        """Computes the net area of each concrete strip."""
        if not self.strain.geometry.section.lBars:
            return np.zeros((n_points, 1))
        
        # Compute height and width of the section
        height = self.strain.geometry.section_height
        width = self.strain.geometry.section_width
        
        # Compute the depth and thickness of the center of each strip
        depth = np.linspace(-height/2 + height/(2*n_points), height/2 - height/(2*n_points), n_points).reshape(-1, 1)
        thk = height / n_points

        # Create concrete strips based on section orientation
        if self.strain.geometry.direction in [BendingDirection.POSITIVE_X, BendingDirection.NEGATIVE_X]:
            concrete_strips = [
                Rectangle(width, thk, (self.strain.geometry.section.shape.center.x, d[0])) for d in depth
            ]
        else:
            concrete_strips = [
                Rectangle(thk, width, (d[0], self.strain.geometry.section.shape.center.y)) for d in depth
            ]

        # Calculate net areas of strips subtracting rebar areas
        bar_shapes = [bar.shape for bar in self.strain.geometry.section.lBars]
        net_areas = np.zeros_like(depth, dtype=float)
        for i, strip in enumerate(concrete_strips):
            net_areas[i] = strip.get_net_area(bar_shapes)
            
        return net_areas
    
    def get_section_internal_force(self, neutral_depth: Union[float, np.ndarray], n_points: int = 100) -> float:
        """Computes the total internal force in the section."""
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        # Compute the total rebar forces
        rebar_forces = self.get_rebar_force(neutral_depth)
        total_rebar_force = rebar_forces.sum(axis=0)

        # Compute the total concrete forces
        concrete_forces = self.get_concrete_force(neutral_depth, n_points)
        total_concrete_force = concrete_forces.sum(axis=0)
        
        return total_rebar_force + total_concrete_force
    
    def get_section_internal_moment(self, neutral_depth: Union[float, np.ndarray], n_points: int = 100) -> float:
        """Computes the total moment in the section."""
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        # Compute the lever arm for the rebars and concrete strips
        rebar_depth = self.strain.geometry.get_bar_depth()
        concrete_depth = np.linspace(
            -self.strain.geometry.section_height/2 + self.strain.geometry.section_height/(2*n_points),
            self.strain.geometry.section_height/2 - self.strain.geometry.section_height/(2*n_points),
            n_points
        ).reshape(-1, 1)
        
        # Compute internal forces in rebars and concrete strips
        rebar_forces = self.get_rebar_force(neutral_depth)
        concrete_forces = self.get_concrete_force(neutral_depth, n_points)

        # Compute the total moment due to rebars and concrete strips
        total_rebar_moment = (rebar_depth.T @ rebar_forces).sum(axis=0)
        total_concrete_moment = (concrete_depth.T @ concrete_forces).sum(axis=0)

        return total_rebar_moment + total_concrete_moment

class UniaixalBending:
    """A class representing uniaxial bending of a rectangular concrete section."""
    
    def __init__(self, sec: RectangleReinforcedSection, bending_direction: str):
        """Initializes the UniaxialBending object.

        Args:
            sec (RectangleReinforcedSection): Concrete section object containing the longitudinal rebars.
            bending_direction (str): Direction of the bending, must be one of:
                'x': when bending vector is in the positive x direction (bending about the x-axis causing compression on the positive y side)
                '-x': when bending vector is in the negative x direction (bending about the x-axis causing compression on the negative y side)
                'y': when bending vector is in the positive y direction (bending about the y-axis causing compression on the positive x side)
                '-y': when bending vector is in the negative y direction (bending about the y-axis causing compression on the negative x side)
        """
        if not isinstance(sec, RectangleReinforcedSection):
            raise TypeError("sec must be an instance of RectangleReinforcedSection.")
        
        self._sec = sec
        self._dir = BendingDirection(bending_direction)
        
        # Initialize analyzers
        self._geometry = GeometryAnalyzer(sec, self._dir)
        self._strain = StrainAnalyzer(self._geometry)
        self._force = ForceAnalyzer(self._strain)
        self._equilibrium = EquilibriumSolver(self._force)
        
    @property
    def sec(self) -> RectangleReinforcedSection:
        """Returns the concrete section."""
        return self._sec
    
    @property
    def dir(self) -> BendingDirection:
        """Returns the bending direction."""
        return self._dir
    
    def get_tensile_rebars(self, neutral_depth: Union[float, np.ndarray]) -> List[LongitudinalRebar]:
        """Returns the list of tensile rebars in the section."""
        if not self.sec.lBars:
            return []
        
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        # Define condition functions for each direction
        conditions = {
            BendingDirection.NEGATIVE_X: lambda bar: bar.center.y < self.sec.shape.y_max - neutral_depth,
            BendingDirection.POSITIVE_X: lambda bar: bar.center.y > self.sec.shape.y_min + neutral_depth,
            BendingDirection.POSITIVE_Y: lambda bar: bar.center.x < self.sec.shape.x_max - neutral_depth,
            BendingDirection.NEGATIVE_Y: lambda bar: bar.center.x > self.sec.shape.x_min + neutral_depth
        }
        
        return [
            bar for bar in self.sec.lBars if conditions[self._dir](bar) and not bar.is_skin_bar
        ]

    def get_compressive_rebars(self, neutral_depth: Union[float, np.ndarray]) -> List[LongitudinalRebar]:
        """Returns the list of compressive rebars in the section."""
        if not self.sec.lBars:
            return []
        
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        tensile_rebars = self.get_tensile_rebars(neutral_depth)
        
        return [bar for bar in self.sec.lBars if bar not in tensile_rebars and not bar.is_skin_bar]
    
    def get_tensile_effective_depth(self, neutral_depth: Union[float, np.ndarray]) -> Optional[float]:
        """Computes the distance from extreme compression fiber to centroid of longitudinal tension reinforcement."""
        if not self.sec.lBars:
            return None
        
        # Compute the coordinates of the extreme compressive fiber
        xFiber_coord = self._geometry.get_extreme_fiber_coord()
        
        # Compute the centroid of the longitudinal tension reinforcement
        tensile_rebars = self.get_tensile_rebars(neutral_depth)
        bar_shapes = [bar.shape for bar in tensile_rebars]
        bar_centroid = get_circle_group_centroid(bar_shapes)
        
        return bar_centroid.distance_to(xFiber_coord)

    def get_compressive_effective_depth(self, neutral_depth: Union[float, np.ndarray]) -> Optional[float]:
        """Computes the distance from extreme compression fiber to centroid of longitudinal compression reinforcement."""
        if not self.sec.lBars:
            return None
        
        if not isinstance(neutral_depth, np.ndarray):   
            neutral_depth = np.array([neutral_depth])

        # Compute the coordinates of the extreme compressive fiber
        xFiber_coord = self._geometry.get_extreme_fiber_coord()
        
        # Compute the centroid of the longitudinal compression reinforcement
        compressive_rebars = self.get_compressive_rebars(neutral_depth)
        bar_shapes = [bar.shape for bar in compressive_rebars]
        bar_centroid = get_circle_group_centroid(bar_shapes)
        
        return bar_centroid.distance_to(xFiber_coord)
    
    def get_balance_point(self, external_force: float) -> Optional[dict]:
        """Computes the balance point of the section."""
        return self._equilibrium.get_balance_point(external_force)
    
    def get_tension_controlled_point(self, external_force: float) -> Optional[dict]:
        """Computes the tension controlled point of the section."""
        return self._equilibrium.get_tension_controlled_point(external_force)
    
    def get_pure_flexure_point(self, external_force: float, config: SolverConfig) -> dict:
        """Computes the pure flexure point of the section."""
        return self._equilibrium.get_pure_flexure_point(external_force, config)

class EquilibriumSolver:
    """Handles solving for neutral axis depth and equilibrium points."""
    
    def __init__(self, force: ForceAnalyzer):
        self._force = force
        
    @property
    def force(self) -> ForceAnalyzer:
        """Returns the force analyzer."""
        return self._force
    
    def solve_neutral_depth_equilibrium(self, external_force: float, config: SolverConfig) -> Tuple[float, float]:
        """Computes the depth of the neutral axis based on the external force."""
        height = self.force.strain.geometry.section_height
        x0 = config.tolerance
        x1 = height - config.tolerance
        
        def equilibrium(neutral_depth: float) -> float:    
            internal_force = self.force.get_section_internal_force(neutral_depth, config.n_discretization_points)
            return internal_force + external_force

        if config.method in ['bisect', 'brentq']:
            result = root_scalar(
                equilibrium,
                bracket=[x0, x1],
                method=config.method,
                xtol=config.tolerance,
                maxiter=config.max_iterations
            )
            
        elif config.method in ['secant', 'newton']:
            result = root_scalar(
                equilibrium,
                method=config.method,
                x0=x0,
                x1=x1,
                xtol=config.tolerance,
                maxiter=config.max_iterations
            )
            
        if result.converged:
            return result.root, config.tolerance
        else:
            raise ValueError("Failed to find a solution for the neutral axis depth.")
    
    def get_neutral_depth_from_strain(self, xBar_strain: float) -> float:
        """Computes the depth of the neutral axis given the strain in the extreme tensile rebar."""
        # Find the depth of the extreme compressive concrete fiber
        xfiber_coord = self.force.strain.geometry.get_extreme_fiber_coord()
        xfiber_depth = xfiber_coord.y if self.force.strain.geometry.direction in [BendingDirection.POSITIVE_X, BendingDirection.NEGATIVE_X] else xfiber_coord.x
        
        # Find depth of the extreme tensile rebars
        xbar_depth = self.force.strain.geometry.get_extreme_bar_depth()
        
        # find concrete ultimate compressive strain (a negative value)
        eps_cu = self.force.strain.geometry.section.mat.eps_u
        
        # Compute the neutral axis depth
        multiplier = 1 if self.force.strain.geometry.direction in [BendingDirection.NEGATIVE_X, BendingDirection.POSITIVE_Y] else -1
        
        return eps_cu / (eps_cu - xBar_strain) * (xfiber_depth - xbar_depth) * multiplier
    
    def get_balance_point(self, external_force: float) -> Optional[dict]:
        """Computes the balance point of the section."""
        if not self.force.strain.geometry.section.lBars:
            return None

        # Find extreme tensile rebars
        xbars = self.force.strain.geometry.get_extreme_bars()

        # Find minimum yield strain among the extreme tensile rebars
        eps_y_min = max([bar.mat.eps_y for bar in xbars])
        
        # Compute the neutral axis depth
        neutral_depth = self.get_neutral_depth_from_strain(eps_y_min)

        # Compute total internal forces
        internal_force = self.force.get_section_internal_force(neutral_depth)
        total_force = internal_force + external_force
        
        # Compute total internal moments
        total_moment = self.force.get_section_internal_moment(neutral_depth)
        
        return {
            'neutral_depth': neutral_depth,
            'force': total_force,
            'moment': total_moment
        }
    
    def get_tension_controlled_point(self, external_force: float) -> Optional[dict]:
        """Computes the tension controlled point of the section."""
        if not self.force.strain.geometry.section.lBars:
            return None
        
        # Strain in the extreme tensile rebar
        eps_s = 0.005
        
        # Compute the neutral axis depth
        neutral_depth = self.get_neutral_depth_from_strain(eps_s)
        
        # Compute total internal forces
        internal_force = self.force.get_section_internal_force(neutral_depth)
        total_force = internal_force + external_force
        
        # Compute total internal moments
        total_moment = self.force.get_section_internal_moment(neutral_depth)
        
        return {
            'neutral_depth': neutral_depth,
            'force': total_force,
            'moment': total_moment
        }
    
    def get_pure_flexure_point(self, external_force: float, config: SolverConfig) -> dict:
        """Computes the pure flexure point of the section."""
        # Solve for the neutral depth at equilibrium with the external force
        neutral_depth, tol = self.solve_neutral_depth_equilibrium(external_force, config)
        
        # Compute total forces in the section
        internal_force = self.force.get_section_internal_force(neutral_depth)
        total_force = internal_force + external_force
        
        # Check if the section is in equilibrium with the external force
        if abs(total_force) > tol:
            raise ValueError("The section is not in equilibrium with the external force.")
    
        # Compute total moments in the section
        total_moment = self.force.get_section_internal_moment(neutral_depth)
        
        return {
            'neutral_depth': neutral_depth,
            'force': total_force,
            'moment': total_moment
        }

# =============================================================        
# Functions
# =============================================================



# def get_section_extreme_fiber(sec: RectangleReinforcedSection, bending_direction: str) -> Point:
#     """Returns the point of extreme fiber for the bending direction in a rectangular concrete section.
#     The reference point is the location of the fiber of maximum strain.

#     Args:
#         sec (RectangleReinforcedSection): Concrete beam section object containing the longitudinal rebars.
#         bending_direction (str): Direction of the bending, must be one of:
#             'x': when bending vector is in the positive x direction (bending about the x-axis causing compression on the negative y side)
#             '-x': when bending vector is in the negative x direction (bending about the x-axis causing compression on the positive y side)
#             'y': when bending vector is in the positive y direction (bending about the y-axis causing compression on the positive x side)
#             '-y': when bending vector is in the negative y direction (bending about the y-axis causing compression on the negative x side)

#     Returns:
#         Point: Reference point for the bending direction.
#     """
#     bending_direction = validate_direction(bending_direction)
#     ref_points = {
#         'x' : Point(sec.shape.center.x, sec.shape.y_min), # compression on -y side
#         '-x': Point(sec.shape.center.x, sec.shape.y_max), # compression on +y side
#         'y' : Point(sec.shape.x_max, sec.shape.center.y), # compression on +x side
#         '-y': Point(sec.shape.x_min, sec.shape.center.y)  # compression on -x side
#     }
    
#     return ref_points.get(bending_direction, None)


# def get_tensile_and_compressive_rebars(sec: RectangleReinforcedSection, neutral_depth: float, bending_direction:str) -> Tuple[List[LongitudinalRebar], List[LongitudinalRebar]]:
    
#     if not sec.lBars:
#         return ([], [])
    
#     bending_direction = validate_direction(bending_direction)
        
#     if bending_direction == '-x':
#         tensile_rebars = [bar for bar in sec.lBars if bar.center.y < sec.y_max - neutral_depth]
#     elif bending_direction == 'x':
#         tensile_rebars = [bar for bar in sec.lBars if bar.center.y > sec.y_min + neutral_depth]
#     elif bending_direction == 'y':
#         tensile_rebars = [bar for bar in sec.lBars if bar.center.x < sec.x_max - neutral_depth]
#     elif bending_direction == '-y':
#         tensile_rebars = [bar for bar in sec.lBars if bar.center.x > sec.x_min + neutral_depth]
    
#     compresive_rebars = [bar for bar in sec.lBars if bar not in tensile_rebars]
    
#     return tensile_rebars, compresive_rebars


    

def nominal_tensile_strength(sec: RectangleReinforcedSection) -> float:
    """Calculates the nominal tensile strength of the concrete section per ACI 318-25: 22.4.3.
    The nominal tensile strength is the sum of the yield strengths of all the rebars in the section.

    Args:
        sec (RectangleReinforcedSection): Concrete section containing longitudinal rebars.

    Returns:
        float: Nominal tensile strength of the section in lbs.
    """
    strength = 0.0
    for rebar in sec.lBars:
        strength += rebar.mat.fy * rebar.area
        
    return strength
        
        
def nominal_compressive_strength(sec: RectangleReinforcedSection) -> float:
    """Calculates the nominal compressive strength of the concrete section per ACI 318-25: 22.4.2..
    The nominal compressive strength is the sum of the compressive strength of the concrete and the yield strengths of all the rebars in the section.

    Args:
        sec (RectangleReinforcedSection): Concrete section containing longitudinal rebars.

    Returns:
        float:  Nominal compressive strength of the section in lbs.
    """
    strength = 0.85 * sec.mat.fc * sec.net_area
    
    for rebar in sec.lBars:
        strength += rebar.mat.fy * rebar.area
    
    return -1 * strength
    

def max_nominal_compressive_strength(sec: RectangleReinforcedSection) -> float:
    """Calculates the maximum nominal compressive strength of the concrete section per ACI 318-25: 22.4.2.1.
    The reduction factor is to account for accidnetal eccentricity in the section.

    Args:
        sec (RectangleReinforcedSection): Concrete section containing longitudinal rebars.

    Returns:
        float: Maximum nominal compressive strength of the section in lbs.
    """
    return 0.80 * nominal_compressive_strength(sec)


def generate_force_moment_interaction(beam: UniaixalBending, external_force: float = 0, n_points:int=100, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    sec = beam.sec
    
    # Create a range of neutral depths from the minimum to the maximum based on the section height
    neutral_depth_range = kwargs.get(
        'neutral_depth_range', 
        (1e-2 * beam.section_height, beam.get_xBar_depth)
    )
    neutral_depth = np.linspace(
        neutral_depth_range[0], 
        neutral_depth_range[1], 
        n_points
    ).reshape(-1, 1)
    
    # Calculate the total internal normal force and moment in the section
    f_total = beam.get_section_internal_force(neutral_depth)
    m_total = beam.get_section_internal_moment(neutral_depth)
   
    # Calculate section forces and moments under pure tension and compression
    tensile_strength = nominal_tensile_strength(sec)
    compressive_strength = nominal_compressive_strength(sec)
    
    # Prepend (np.inf, tensile_strength, 0)
    neutral_depth = np.vstack(([np.inf], neutral_depth))
    f_total = np.hstack(([tensile_strength], f_total))
    m_total = np.hstack(([0], m_total))

    # Append (np.inf, compressive_strength, 0)
    neutral_depth = np.vstack((neutral_depth, [np.inf]))
    f_total = np.hstack((f_total, [compressive_strength]))
    m_total = np.hstack((m_total, [0]))
    
    return neutral_depth, f_total, m_total


    





    
    
    
    
    
# Main function =========================================================      
def main():
    pass
    
    
if __name__ == "__main__":
    main()