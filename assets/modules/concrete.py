################################
# Concrete Section Module
# Written by: Hossein Karagah
# Date: 2025-03-10
# Description: This module provides classes and functions to model and analyze rectangular concrete sections with rebars.
################################

from pickletools import read_bytes1
from re import U
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

# Main Classes ################################################

class BendingDirection(Enum):
    """Enumeration for bending directions."""
    POSITIVE_X = "x"
    NEGATIVE_X = "-x"
    POSITIVE_Y = "y"
    NEGATIVE_Y = "-y"

    def get_primary_axis(self) -> str:
        """Returns the primary axis for the bending direction."""
        return 'y' if self in [self.POSITIVE_X, self.NEGATIVE_X] else 'x'


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
        self._long_bars: List[LongitudinalRebar] = []
        self._trans_bars: List[TransverseRebar] = []   
        
    @property
    def long_bars(self) -> List[LongitudinalRebar]:
        """Returns the list of longitudinal rebars in the beam section."""
        return self._long_bars
    
    @property
    def trans_bars(self) -> List[TransverseRebar]:
        """Returns the list of transverse rebars (stirrups) in the beam section."""
        return self._trans_bars

    def add_trans_bar(self, trans_bar: TransverseRebar):
        """Adds a transverse rebar (stirrup) to the concrete section.
        The transverse rebar is added to the list of transverse bars in the section.
        A rectangular section can have multiple transverse rebars, and each can be looped or single legged.

        Args:
            bar (TransverseRebar): Transverse rebar object to be added.
            
        """
        if not isinstance(trans_bar, TransverseRebar):
            raise TypeError("trans_bar must be an instance of TransverseRebar.")
        self._trans_bars.append(deepcopy(trans_bar))
        
    def add_long_bar(self, long_bar: LongitudinalRebar, center: Union[Point, Tuple[float, float]]):
        """Adds a rebar to the wall.
        The rebar is added at the specified center point. The function checks if the new rebar overlaps with existing rebars and if it is within the wall bounds minus cover.

        Args:
            long_bar (LongitudinalRebar): Longitudinal reinforcement object to be added.
            center (Union[Point, Tuple[float, float]]): Center point of the longitudinal rebar.
                If a tuple is provided, it will be converted to a Point object.

        Raises:
            ValueError: If the new rebar overlaps with existing rebars or if it is outside the wall bounds minus cover.
            ValueError: If the rebar is outside the wall bounds minus cover.
        """
        if isinstance(center, tuple):
            center = Point(center[0], center[1])
        
        bar = deepcopy(long_bar)
        bar.center = center
        
        # Check if new rebar overlaps with existing rebars
        for existing_rebar in self.long_bars:
            if existing_rebar.shape.overlapped_with(bar.shape):
                raise ValueError(f"Rebar at {bar.center} overlaps with existing rebar at {existing_rebar.center}.")
        
        # Check if new rebar is within core region
        if not bar.shape.within(self.core()):
            x_min, x_max, y_min, y_max = self._get_bounds(self.core())
            raise ValueError(f"Rebar at {bar.center} is outside the core region x-range:({x_min}, {x_max}), y-range: ({y_min}, {y_max}).")
            
        self._long_bars.append(bar)

    def add_long_bar_row(self, long_bar: LongitudinalRebar, center: Union[Point, Tuple[float, float]], 
                    n: int, direction: Tuple[float, float], spacing: Optional[float] = None):
        """Adds a row of rebars to the wall section.
        
        Args:
            long_bar (LongitudinalRebar): Longitudinal reinforcement object to be added.
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
            spacing = self._calculate_auto_spacing(center, dx, dy, n, long_bar)
        
        if spacing <= 0:
            raise ValueError("Invalid spacing: too small or negative.")
        
        # Generate and add rebars
        positions = self._generate_rebar_positions(center, dx, dy, n, spacing)
        for pos in positions:
            self.add_long_bar(long_bar, pos)

    def _calculate_auto_spacing(self, center: Point, dx: float, dy: float, n: int, long_bar: LongitudinalRebar) -> float:
        """Calculate automatic spacing based on available space in core region."""
        x_min, x_max, y_min, y_max = self._get_bounds(self.core(long_bar))
        
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
        for bar in self.long_bars:
            bar._shape.translate(translation)
    
    def rotate(self, angle_deg: float, pivot: Union[Point, Tuple[float, float]] = None):
        """Rotates the section by angle_deg around the pivot point."""
        self._shape.rotate(angle_deg, pivot)
        for bar in self.long_bars:
            bar._shape.rotate(angle_deg, pivot=self._shape.center) 
            
    @property
    def long_bar_x_max(self):
        """Returns the maximum x coordinate of all rebars in the section."""
        if not self.long_bars:
            return None
        return max([rebar.center.x for rebar in self.long_bars])
    
    @property
    def long_bar_x_min(self):
        """Returns the minimum x coordinate of all rebars in the section."""
        if not self.long_bars:
            return None
        return min([rebar.center.x for rebar in self.long_bars])
    
    @property
    def long_bar_y_max(self):
        """Returns the maximum y coordinate of all rebars in the section."""
        if not self.long_bars:
            return None
        return max([rebar.center.y for rebar in self.long_bars])
    
    @property
    def long_bar_y_min(self):
        """Returns the minimum y coordinate of all rebars in the section."""
        if not self.long_bars:
            return None
        return min([rebar.center.y for rebar in self.long_bars])
            

        
    @property
    def trans_bar_max_dia(self) -> float:
        """Returns the maximum diameter of transverse bars in the section if looped is True.
        It does not include single-legged transverse bars since they do not enclose the section.
        """
        if not self.trans_bars:
            return 0.0
        else:
            return max(bar.dia if bar.is_looped else 0 for bar in self.trans_bars)
        
    def core(self, rebar: Optional[LongitudinalRebar] = None) -> Rectangle:
        """Computes the concrete core excluding the cover, transverse bars, and optionally a rebar.
        The core is the area where the main rebars can be placed without overlapping with the cover or transverse bars.

        Args:
            rebar (Optional[RebarSection], optional): longitudinal rabar. Defaults to None.

        Returns:
            sh.Rectangle: The core rectangle shape that defines the area where longitudinal rebars can be placed.
        """
        offset = 0 if rebar is None else rebar.dia            
        width = self.shape.width - 2 * self.cover - 2 * self.trans_bar_max_dia - offset
        height = self.shape.height - 2 * self.cover - 2 * self.trans_bar_max_dia - offset
        
        return Rectangle(width, height, self._shape.center, self._shape.rotation) 
    
    @property
    def net_area(self) -> float:
        """Returns the net area of the concrete section excluding the area occupied by longitudinal rebars."""
        long_bar_area = sum([bar.area for bar in self.long_bars])
        return self.gross_area - long_bar_area 


class GeometryAnalyzer:
    def __init__(self, section: RectangleReinforcedSection, direction: BendingDirection):
        """
        Args:
            section (RectangleReinforcedSection): The concrete section.
            direction (BendingDirection): The bending direction (must be an enum, not a string).
        """
        self.section = section
        self.direction = direction
    def get_xFiber_coord(self):
        ref_points = {
            BendingDirection.POSITIVE_X: Point(self.section.shape.center.x, self.section.shape.y_min),
            BendingDirection.NEGATIVE_X: Point(self.section.shape.center.x, self.section.shape.y_max),
            BendingDirection.POSITIVE_Y: Point(self.section.shape.x_max, self.section.shape.center.y),
            BendingDirection.NEGATIVE_Y: Point(self.section.shape.x_min, self.section.shape.center.y)
        }
        return ref_points.get(self.direction, None)
    def section_height(self):
        if self.direction in [BendingDirection.POSITIVE_X, BendingDirection.NEGATIVE_X]:
            return self.section.shape.y_max - self.section.shape.y_min
        else:
            return self.section.shape.x_max - self.section.shape.x_min
    def section_width(self):
        if self.direction in [BendingDirection.POSITIVE_X, BendingDirection.NEGATIVE_X]:
            return self.section.shape.x_max - self.section.shape.x_min
        else:
            return self.section.shape.y_max - self.section.shape.y_min
        
    def get_trans_bar_area(self):
        """Returns the total area of transverse bars in the section."""
        if not self.section.trans_bars:
            return 0.0
        if self.direction in [BendingDirection.POSITIVE_X, BendingDirection.NEGATIVE_X]:
            return sum(bar.Ax for bar in self.section.trans_bars)
        else:
            return sum(bar.Ay for bar in self.section.trans_bars)
    def get_concrete_strip_depth(self, n_disc=100):
        height = self.section_height()
        strip_thk = height / n_disc
        return np.linspace(-height/2 + strip_thk/2, height/2 - strip_thk/2, n_disc).reshape(-1, 1)
    def get_concrete_strip_net_area(self, n_disc=100):
        if not self.section.long_bars:
            return np.zeros((n_disc, 1))
        height = self.section_height()
        width = self.section_width()
        depth = self.get_concrete_strip_depth(n_disc)
        thk = height / n_disc
        if self.direction in [BendingDirection.POSITIVE_X, BendingDirection.NEGATIVE_X]:
            concrete_strips = [Rectangle(width, thk, (self.section.shape.center.x, d[0])) for d in depth]
        else:
            concrete_strips = [Rectangle(thk, width, (d[0], self.section.shape.center.y)) for d in depth]
        bar_shapes = [bar.shape for bar in self.section.long_bars]
        net_areas = np.zeros(len(concrete_strips), dtype=float)
        for i, strip in enumerate(concrete_strips):
            net_areas[i] = strip.get_net_area(bar_shapes)
        return net_areas.reshape(-1, 1)

class StrainAnalyzer:
    def __init__(self, section: RectangleReinforcedSection, direction: BendingDirection):
        """
        Args:
            section (RectangleReinforcedSection): The concrete section.
            direction (BendingDirection): The bending direction (must be an enum, not a string).
        """
        self.section = section
        self.direction = direction
    def _depth_multiplier(self):
        return 1 if self.direction in [BendingDirection.NEGATIVE_X, BendingDirection.POSITIVE_Y] else -1
    
    def get_curvature(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        try:
            return np.divide(self.section.mat.eps_u, neutral_depth)
        except ZeroDivisionError:
            raise ValueError("Neutral depth cannot be zero.")
        
    def get_long_bar_depth(self) -> np.ndarray:
        if not self.section.long_bars:
            return np.array([])
        if self.direction in [BendingDirection.POSITIVE_X, BendingDirection.NEGATIVE_X]:
            return np.array([bar.center.y for bar in self.section.long_bars]).reshape(-1, 1)
        else:
            return np.array([bar.center.x for bar in self.section.long_bars]).reshape(-1, 1)
        
    def get_rebar_strain(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
        if not self.section.long_bars:
            return np.array([])
        neutral_depth = np.atleast_2d(neutral_depth).astype(float)
        if neutral_depth.shape[1] != 1:
            neutral_depth = neutral_depth.reshape(-1, 1)
        if np.any(np.isnan(neutral_depth)):
            raise ValueError("neutral_depth contains NaNs")
        curv = self.get_curvature(neutral_depth)
        sec_height = GeometryAnalyzer(self.section, self.direction).section_height()
        bar_depth = self.get_long_bar_depth()
        mi = self._depth_multiplier()
        
        # # Diagnostic print statements
        # print("curv:", curv)
        # print("bar_depth:", bar_depth)
        # print("mi:", mi)
        # print("sec_height:", sec_height)

        return curv.T * (bar_depth * mi - (sec_height / 2 - neutral_depth.T))
    
    def get_rebar_stress(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
        if not self.section.long_bars:
            return np.array([])
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        strain = self.get_rebar_strain(neutral_depth)
        return np.array([bar.mat.stress(eps) for bar, eps in zip(self.section.long_bars, strain)])

    def get_concrete_strain(self, neutral_depth: Union[float, np.ndarray], n_disc: int = 100) -> np.ndarray:
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        curv = self.get_curvature(neutral_depth)
        height = GeometryAnalyzer(self.section, self.direction).section_height()
        depth = GeometryAnalyzer(self.section, self.direction).get_concrete_strip_depth(n_disc)
        mi = self._depth_multiplier()
        return curv.T * (depth * mi - (height / 2 - neutral_depth.T))
    
    def get_concrete_stress(self, neutral_depth: Union[float, np.ndarray], n_disc: int = 100) -> np.ndarray:
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        strain = self.get_concrete_strain(neutral_depth, n_disc)
        return self.section.mat.stress(strain)

class ForceAnalyzer:
    def __init__(self, section: RectangleReinforcedSection, direction: BendingDirection):
        """
        Args:
            section (RectangleReinforcedSection): The concrete section.
            direction (BendingDirection): The bending direction (must be an enum, not a string).
        """
        self.section = section
        self.direction = direction
        
    def get_tensile_rebars(self, neutral_depth: Optional[Union[float, list, np.ndarray]]):
        if not self.section.long_bars:
            return []
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        conditions = {
            BendingDirection.NEGATIVE_X: lambda bar: bar.center.y < self.section.shape.y_max - neutral_depth,
            BendingDirection.POSITIVE_X: lambda bar: bar.center.y > self.section.shape.y_min + neutral_depth,
            BendingDirection.POSITIVE_Y: lambda bar: bar.center.x < self.section.shape.x_max - neutral_depth,
            BendingDirection.NEGATIVE_Y: lambda bar: bar.center.x > self.section.shape.x_min + neutral_depth
        }
        return [bar for bar in self.section.long_bars if conditions[self.direction](bar) and not bar.is_skin_bar]
    
    def get_compressive_rebars(self, neutral_depth: Optional[Union[float, list, np.ndarray]]):
        if not self.section.long_bars:
            return []
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        tensile_rebars = self.get_tensile_rebars(neutral_depth)
        return [bar for bar in self.section.long_bars if bar not in tensile_rebars and not bar.is_skin_bar]
    
    def get_rebar_force(self, neutral_depth):
        if not self.section.long_bars:
            return np.array([])
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        stress = StrainAnalyzer(self.section, self.direction).get_rebar_stress(neutral_depth)
        return np.array([bar.area * s for bar, s in zip(self.section.long_bars, stress)])
    
    def get_tensile_effective_depth(self, neutral_depth: Optional[Union[float, list, np.ndarray]]):
        if not self.section.long_bars:
            return None
        xFiber_coord = GeometryAnalyzer(self.section, self.direction).get_xFiber_coord()
        tensile_rebars = self.get_tensile_rebars(neutral_depth)
        bar_shapes = [bar.shape for bar in tensile_rebars]
        bar_centroid = get_circle_group_centroid(bar_shapes)
        return bar_centroid.distance_to(xFiber_coord)
    
    def get_compressive_effective_depth(self, neutral_depth: Optional[Union[float, list, np.ndarray]]):
        if not self.section.long_bars:
            return None
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        xFiber_coord = GeometryAnalyzer(self.section, self.direction).get_xFiber_coord()
        compressive_rebars = self.get_compressive_rebars(neutral_depth)
        bar_shapes = [bar.shape for bar in compressive_rebars]
        bar_centroid = get_circle_group_centroid(bar_shapes)
        return bar_centroid.distance_to(xFiber_coord)
    
    def get_xBar_depth(self, neutral_depth: Optional[Union[float, list, np.ndarray]] = None, section_height: Optional[float] = None) -> Union[None, np.ndarray]:
        if not self.section.long_bars:
            return None
        try:
            if neutral_depth is None or (isinstance(neutral_depth, (float, int)) and neutral_depth == 0):
                # No neutral depth provided, return the extreme value for the direction
                if self.direction == BendingDirection.NEGATIVE_X:
                    return np.array([min([bar.center.y for bar in self.section.long_bars])])
                elif self.direction == BendingDirection.POSITIVE_X:
                    return np.array([max([bar.center.y for bar in self.section.long_bars])])
                elif self.direction == BendingDirection.POSITIVE_Y:
                    return np.array([min([bar.center.x for bar in self.section.long_bars])])
                elif self.direction == BendingDirection.NEGATIVE_Y:
                    return np.array([max([bar.center.x for bar in self.section.long_bars])])
                else:
                    return None
            else:
                if not section_height:
                    section_height = GeometryAnalyzer(self.sec, self.dir).section_height()
                if not isinstance(neutral_depth, np.ndarray):
                    neutral_depth = np.array(neutral_depth, ndmin=1)
                neutral_coord = section_height / 2 - neutral_depth
                min_depth = np.full(neutral_coord.shape, np.nan)
                for i, coord in enumerate(neutral_coord):
                    if self.direction == BendingDirection.NEGATIVE_X:
                        candidates = [bar.center.y for bar in self.section.long_bars if bar.center.y < coord]
                        min_depth[i] = min(candidates) if candidates else np.nan
                    elif self.direction == BendingDirection.POSITIVE_X:
                        candidates = [bar.center.y for bar in self.section.long_bars if bar.center.y > coord]
                        min_depth[i] = max(candidates) if candidates else np.nan
                    elif self.direction == BendingDirection.POSITIVE_Y:
                        candidates = [bar.center.x for bar in self.section.long_bars if bar.center.x < coord]
                        min_depth[i] = min(candidates) if candidates else np.nan
                    elif self.direction == BendingDirection.NEGATIVE_Y:
                        candidates = [bar.center.x for bar in self.section.long_bars if bar.center.x > coord]
                        min_depth[i] = max(candidates) if candidates else np.nan
                    else:
                        min_depth[i] = np.nan
                return min_depth
        except ValueError:
            return None
    
    def get_xBars(self, neutral_depth: Optional[Union[float, np.ndarray]] = None, section_height: Optional[float] = None) -> Union[None, list]:
        if not self.section.long_bars:
            return []
        depth = self.get_xBar_depth(neutral_depth, section_height)
        if depth is None:
            return []
        xBars = []
        for d in depth:
            if self.direction in [BendingDirection.POSITIVE_X, BendingDirection.NEGATIVE_X]:
                xBars.append([bar for bar in self.section.long_bars if bar.center.y == d])
            else:
                xBars.append([bar for bar in self.section.long_bars if bar.center.x == d])
        return xBars
        
    def get_concrete_force(self, neutral_depth: Union[float, np.ndarray], n_disc: int = 100) -> np.ndarray:
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        net_areas = GeometryAnalyzer(self.section, self.direction).get_concrete_strip_net_area(n_disc)
        stress = StrainAnalyzer(self.section, self.direction).get_concrete_stress(neutral_depth, n_disc)
        return stress * net_areas


@dataclass
class SolverConfig:
    method: str = 'secant'
    x0: float = 0.1
    x1: float = 0.9
    tolerance: float = 1e-2
    max_iterations: int = 100
    n_disc: int = 100
    

class EquilibriumSolver:
    def __init__(self, section: RectangleReinforcedSection, bending_direction: BendingDirection):
        self.section = section
        if isinstance(bending_direction, str):
            try:
                self.direction = BendingDirection(bending_direction)
            except ValueError:
                raise ValueError(f"Invalid direction string: {bending_direction}")
        elif isinstance(bending_direction, BendingDirection):
            self.direction = bending_direction
        else:
            raise TypeError("'direction' must be a str or 'BendingDirection'")
    def solve_neutral_depth_equilibrium(self, external_force: float, config: SolverConfig) -> float:
        """
        Computes the neutral axis depth at which the section is in equilibrium with the given external force.
        Uses the configuration from SolverConfig for method, tolerance, max iterations, and discretization points.
        Returns a dict with 'neutral_depth', 'force', and 'moment'.
        """

        force_analyzer = ForceAnalyzer(self.section, self.direction)
        
        height = GeometryAnalyzer(self.section, self.direction).section_height()
        
        def equilibrium(neutral_depth: float) -> float:
            # Compute total internal force at given neutral axis depth
            rebar_force = force_analyzer.get_rebar_force(neutral_depth)
            concrete_force = force_analyzer.get_concrete_force(neutral_depth, config.n_disc)
            total_internal_force = np.sum(rebar_force) + np.sum(concrete_force)
            return total_internal_force - external_force
        
        method = config.method
        x0 = config.x0 * height
        x1 = config.x1 * height
        tol = config.tolerance
        max_iter = config.max_iterations
        
        if method in ['bisect', 'brentq']:
            result = root_scalar(equilibrium, bracket=[x0, x1], method=method, xtol=tol, maxiter=max_iter)
        elif method in ['secant', 'newton']:
            result = root_scalar(equilibrium, method=method, x0=x0, x1=x1, xtol=tol, maxiter=max_iter)
        else:
            raise ValueError(f"Unknown root-finding method: {method}")
        
        if not result.converged:
            raise ValueError("Failed to find a solution for the neutral axis depth.")
        
        return np.array([result.root])


class UniaxialBending:
    def __init__(self, sec: RectangleReinforcedSection, bending_direction):
        if not isinstance(sec, RectangleReinforcedSection):
            raise TypeError("'sec' must be an instance of 'RectangleReinforcedSection'.")
        if isinstance(bending_direction, str):
            try:
                self._dir = BendingDirection(bending_direction.lower())
            except ValueError:
                raise ValueError(f"Invalid bending_direction string: {bending_direction}")
        elif isinstance(bending_direction, BendingDirection):
            self._dir = bending_direction
        else:
            raise TypeError("'bending_direction' must be a 'str' or 'BendingDirection'.")
        self._sec = sec
        self.geometry = GeometryAnalyzer(sec, self._dir)
        self.strain = StrainAnalyzer(sec, self._dir)
        self.force = ForceAnalyzer(sec, self._dir)
        self.equilibrium = EquilibriumSolver(sec, self._dir)
        
    @property
    def sec(self):
        return self._sec
    
    @property
    def dir(self):
        return self._dir
    
    def depth_multiplier(self):
        return self.strain._depth_multiplier()

    def get_xFiber_coord(self):
        return self.geometry.get_xFiber_coord()
    
    def section_height(self):
        return self.geometry.section_height()
    
    def section_width(self):
        return self.geometry.section_width()
    
    def get_trans_bar_area(self):
        return self.geometry.get_trans_bar_area()
    
    def get_concrete_strip_depth(self, n_disc=100):
        return self.geometry.get_concrete_strip_depth(n_disc)
    
    def get_concrete_strip_net_area(self, n_disc=100):
        return self.geometry.get_concrete_strip_net_area(n_disc)
    
    def get_curvature(self, neutral_depth):
        return self.strain.get_curvature(neutral_depth)
    
    def get_long_bar_depth(self):
        return self.strain.get_long_bar_depth()
    
    def get_rebar_strain(self, neutral_depth):
        return self.strain.get_rebar_strain(neutral_depth)
    
    def get_rebar_stress(self, neutral_depth):
        return self.strain.get_rebar_stress(neutral_depth)
    
    def get_tensile_rebars(self, neutral_depth):
        return self.force.get_tensile_rebars(neutral_depth)
    
    def get_compressive_rebars(self, neutral_depth):
        return self.force.get_compressive_rebars(neutral_depth)
    
    def get_rebar_force(self, neutral_depth):
        return self.force.get_rebar_force(neutral_depth)
    
    def get_tensile_effective_depth(self, neutral_depth):
        return self.force.get_tensile_effective_depth(neutral_depth)
    
    def get_compressive_effective_depth(self, neutral_depth):
        return self.force.get_compressive_effective_depth(neutral_depth)
    
    def get_xBar_depth(self, neutral_depth: Optional[Union[float, np.ndarray]] = None) -> Union[np.ndarray, None]:
        return self.force.get_xBar_depth(neutral_depth, self.section_height())
    
    def get_xBars(self, neutral_depth: Optional[Union[float, np.ndarray]] = None) -> Union[None, list]:
        return self.force.get_xBars(neutral_depth, self.section_height())
    
    def get_concrete_strain(self, neutral_depth, n_disc=1000):
        return self.strain.get_concrete_strain(neutral_depth, n_disc)
    
    def get_concrete_stress(self, neutral_depth, n_disc=1000):
        return self.strain.get_concrete_stress(neutral_depth, n_disc)
    
    def get_concrete_force(self, neutral_depth, n_disc=1000):
        return self.force.get_concrete_force(neutral_depth, n_disc)

    # """A class representing uniaxial bending of a rectangular concrete section.
    # This class is used to compute the bending capacity of a concrete section under uniaxial bending.
    # """
    # def __init__(self, sec: RectangleReinforcedSection, bending_direction: str):
    #     """Initializes the UniaxialBending object.

    #     Args:
    #         con (RectangleReinforcedSection): Concrete section object containing the longitudinal rebars.
    #         bending_direction (str): Direction of the bending, must be one of:
    #             'x': when bending vector is in the positive x direction (bending about the x-axis causing compression on the positive y side)
    #             '-x': when bending vector is in the negative x direction (bending about the x-axis causing compression on the negative y side)
    #             'y': when bending vector is in the positive y direction (bending about the y-axis causing compression on the positive x side)
    #             '-y': when bending vector is in the negative y direction (bending about the y-axis causing compression on the negative x side)

    #     Raises:
            
    #     """
        
    #     if not isinstance(sec, RectangleReinforcedSection):
    #         raise TypeError("sec must be an instance of RectangleReinforcedSection.")
        
    #     self._sec = sec
    #     self._dir = self._validate_direction(bending_direction)
        
    # @property
    # def sec(self) -> RectangleReinforcedSection:
    #     """Returns the concrete section."""
    #     return self._sec
    
    # @property
    # def dir(self) -> str:
    #     """Returns the bending direction."""
    #     return self._dir
    
    # @property
    # def _depth_multiplier(self) -> int:
    #     """Returns the multiplier for the depth depending on the bending direction.

    #     Returns:
    #         int: 1 if the '-x' and '+y' bending direction, -1 if the bending direction is '+x' or '-y'.
    #     """
    #     return 1 if self._dir in ['-x', 'y'] else -1
    
    # @staticmethod
    # def _validate_direction(direction: str) -> str:
    #     """Checks if the given direction is valid.
        
    #     Args:
    #         direction (str): Direction string to check.
        
    #     Raises:
    #         TypeError: If the direction is not a string.
    #         ValueError: If the direction is None or not one of the valid options.
    #     """
    #     valid_directions = ['x', '-x', 'y', '-y']
        
    #     if not isinstance(direction, str):
    #         raise TypeError("Direction must be a string.")
        
    #     if direction is None:
    #         raise ValueError("Direction cannot be None.")
        
    #     direction = direction.lower()
    #     if direction not in valid_directions:
    #         raise ValueError(f"Direction must be one of {valid_directions}.")
        
    #     return direction
    
    # @property
    # def net_area(self) -> float:
    #     """Returns the net area of the section."""
    #     long_bar_area = sum([bar.area for bar in self.sec.long_bar])
    #     return self.sec.gross_area - long_bar_area

    # @property
    # def trans_bar_area(self) -> float:
    #     """Returns the total area of transverse bars in the section."""
    #     if not self.sec.trans_bars:
    #         return 0.0
        
    #     if self._dir in ['x', '-x']:
    #         return sum(bar.Ax for bar in self.sec.trans_bars)
    #     elif self._dir in ['y', '-y']:
    #         return sum(bar.Ay for bar in self.sec.trans_bars) 

    # def get_tensile_rebars(self, neutral_depth: Union[float, np.ndarray]) -> List[LongitudinalRebar]:
    #     """Returns the list of tensile rebars in the section based on the bending direction and neutral depth. The tensile rebars are those that are located in the tension zone of the section and are not marked as skin rebars.

    #     Args:
    #         neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

    #     Returns:
    #         List[LongitudinalRebar]: Returns the list of tensile rebars in the section based on the bending direction and neutral depth. Returns an empty list if no rebars are defined.
    #     """
    #     if not self.sec.long_bar:
    #         return []
        
    #     if not isinstance(neutral_depth, np.ndarray):
    #         neutral_depth = np.array([neutral_depth])
        
    #     # Define condition functions for each direction
    #     conditions:dict[str, Callable[[LongitudinalRebar], bool]] = {
    #         '-x': lambda bar: bar.center.y < self.sec.shape.y_max - neutral_depth,
    #         'x': lambda bar: bar.center.y > self.sec.shape.y_min + neutral_depth,
    #         'y': lambda bar: bar.center.x < self.sec.shape.x_max - neutral_depth,
    #         '-y': lambda bar: bar.center.x > self.sec.shape.x_min + neutral_depth
    #     }
        
    #     return [
    #         bar for bar in self.sec.long_bar if conditions[self._dir](bar) and not bar.is_skin_bar
    #     ]

    # def get_compressive_rebars(self, neutral_depth: Union[float, np.ndarray]) -> List[LongitudinalRebar]:
    #     """Returns the list of compressive rebars in the section based on the bending direction and neutral depth. The compressive rebars are those that are located in the compression zone of the section and are not marked as skin rebars.

    #     Args:
    #         neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

    #     Returns:
    #         List[LongitudinalRebar]: Returns the list of compressive rebars in the section based on the bending direction and neutral depth. Returns an empty list if no rebars are defined.
    #     """
        
    #     if not self.sec.long_bar:
    #         return []
        
    #     if not isinstance(neutral_depth, np.ndarray):
    #         neutral_depth = np.array([neutral_depth])
        
    #     tensile_rebars = self.get_tensile_rebars(neutral_depth)
        
    #     return [bar for bar in self.sec.long_bar if bar not in tensile_rebars and not bar.is_skin_bar]
    
    # @property
    # def get_xFiber_coord(self) -> Point:
    #     """Returns the coordinates of the extreme compressive fiber for the bending direction in a rectangular concrete section. The point of extreme compressive fiber is the location of the fiber of maximum compressive strain.

    #     Returns:
    #         Point: Point of extreme compressive fiber considering the bending direction.
    #     """
    #     ref_points = {
    #         'x' : Point(self.sec.shape.center.x, self.sec.shape.y_min),
    #         '-x': Point(self.sec.shape.center.x, self.sec.shape.y_max),
    #         'y' : Point(self.sec.shape.x_max, self.sec.shape.center.y),
    #         '-y': Point(self.sec.shape.x_min, self.sec.shape.center.y)
    #     }
        
    #     return ref_points.get(self._dir, None)
    
    # def get_tensile_effective_depth(self, neutral_depth: Union[float, np.ndarray]):
    #     """Computes the distance from extreme compression fiber to centroid of longitudinal tension reinforcement. If no rebar is defined, it returns None.
        
    #     Args:
    #         neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

    #     Returns:
    #         float: Distance from extreme compression fiber to centroid of tension reinforcement.
    #         None: If no rebars are defined.

    #     """
        
    #     if not self.sec.long_bar:
    #         return None
        
    #     # Compute the coornidates of the extreme compressive fiber based on the bending direction
    #     xFiber_coord = self.get_xFiber_coord
        
    #     # Compute the centroid of the longitudinal tension reinforcement based on the bending direction
    #     tensile_rebars = self.get_tensile_rebars(neutral_depth)
    #     bar_shapes = [bar.shape for bar in tensile_rebars]
    #     bar_centroid = shapes.get_circle_group_centroid(bar_shapes)
        
    #     return bar_centroid.distance_to(xFiber_coord)

    # def get_compressive_effective_depth(self, neutral_depth: Union[float, np.ndarray]):
    #     """Computes the distance from extreme compression fiber to centroid of longitudinal compression reinforcement. If no rebar is defined, it returns None.
        
    #     Args:
    #         neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

    #     Returns:
    #         float: Distance from extreme compression fiber to centroid of tension reinforcement.
    #         None: If no rebars are defined.
    #     """
        
    #     if not self.sec.long_bar:
    #         return None
        
    #     if not isinstance(neutral_depth, np.ndarray):   
    #         neutral_depth = np.array([neutral_depth])

    #     # Compute the coornidates of the extreme compressive fiber based on the bending direction
    #     xFiber_coord = self.get_xFiber_coord
        
    #     # Compute the centroid of the longitudinal compression reinforcement based on the bending direction
    #     compressive_rebars = self.get_compressive_rebars(neutral_depth)
    #     bar_shapes = [bar.shape for bar in compressive_rebars]
    #     bar_centroid = shapes.get_circle_group_centroid(bar_shapes)
        
    #     return bar_centroid.distance_to(xFiber_coord)
            
    #     # if bending_direction == 'x':
    #     #     return abs(self.x_max - self.long_bar_x_max)
    #     # elif bending_direction == '-x':
    #     #     return abs(self.x_min - self.long_bar_x_min)
    # def get_xBar_depth(self):
    #     if not self._sec.long_bar:
    #         return None
    #     min_depth = {
    #         BendingDirection.X: max([bar.center.y for bar in self._sec.long_bar]),
    #         BendingDirection.NEG_X: min([bar.center.y for bar in self._sec.long_bar]),
    #         BendingDirection.Y: min([bar.center.x for bar in self._sec.long_bar]),
    #         BendingDirection.NEG_Y: max([bar.center.x for bar in self._sec.long_bar])
    #     }
    #     return min_depth[self._dir]
        
    #     if not self.sec.long_bar:
    #         return []

    #     # Find the depth of the outer most tensile rebar based on the bending direction
    #     depth = self.get_xBar_depth
        
    #     # Find the rebars at the outer most depth based on the bending direction
    #     if self._dir in ['x', '-x']:
    #         return [bar for bar in self.sec.long_bar if bar.center.y == depth]
    #     else:
    #         return [bar for bar in self.sec.long_bar if bar.center.x == depth]        

    # @property
    # def get_long_bar_depth(self) -> np.ndarray:
    #     """Returns the depth of each rebar from the extreme compressive fiber based on the bending direction.

    #     Returns:
    #         np.ndarray: Returns the depth of each rebar from the extreme compressive fiber based on the bending direction. Returns an empty array if no rebars are defined.
    #     """
        
    #     if not self.sec.long_bar:
    #         return np.array([])
        
    #     # Compute the depth of each rebar based on bending direction
    #     if self._dir in ['x', '-x']:
    #         return np.array([bar.center.y for bar in self.sec.long_bar]).reshape(-1, 1)
    #     else: # bending_direction in ['y', '-y']
    #         return np.array([bar.center.x for bar in self.sec.long_bar]).reshape(-1, 1)  
        
    # @property
    # def section_height(self) -> float:
    #     """Returns the height of the section based on the bending direction."""
    #     if self._dir in ['x', '-x']:
    #         return self.sec.shape.y_max - self.sec.shape.y_min
    #     else: # bending_direction in ['y', '-y']
    #         return self.sec.shape.x_max - self.sec.shape.x_min
        
    # @property
    # def section_width(self) -> float:
    #     """Returns the width of the section based on the bending direction."""
    #     if self._dir in ['x', '-x']:
    #         return self.sec.shape.x_max - self.sec.shape.x_min
    #     else: # bending_direction in ['y', '-y']
    #         return self.sec.shape.y_max - self.sec.shape.y_min

    # def get_curvature(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
    #     """Returns the curvature of the section based on the bending direction and neutral depth.
    #     The curvature is computed as the ratio of the ultimate compressive strain in concrete to the depth of the neutral axis.

    #     Args:
    #         neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

    #     Returns:
    #         np.ndarray: Returns the curvature of the section based on the bending direction and neutral depth.
            
    #     Raises:
    #         ValueError: If neutral_depth is zero.
    #     """
        
    #     if not isinstance(neutral_depth, np.ndarray):
    #         neutral_depth = np.array([neutral_depth])
        
    #     try:
    #         return np.divide(self.sec.mat.eps_u, neutral_depth)
    #     except ZeroDivisionError:
    #         raise ValueError("Neutral depth cannot be zero.")

    # def get_rebar_strain(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
    #     """Computes the strain in the longitudinal rebars based on the bending direction and neutral depth.

    #     Args:
    #         neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

    #     Returns:
    #         np.ndarray: Returns the strain in the longitudinal rebars based on the bending direction and neutral depth. Returns an empty array if no rebars are defined.
    #     """
    #     if not self.sec.long_bar:
    #         return np.array([])
        
    #     if not isinstance(neutral_depth, np.ndarray):
    #         neutral_depth = np.array([neutral_depth])
        
    #     # Compute curvature based on neutral depth
    #     curv = self.get_curvature(neutral_depth)

    #     # Compute total depth of the section considering the bending direction
    #     sec_height = self.section_height
        
    #     # Compute the depth of each rebar relative to the section centroid based on bending direction
    #     bar_depth = self.get_long_bar_depth 

    #     # Adjust depth based on bending direction
    #     mi = self._depth_multiplier

    #     # Calculate strain, stress, and force for each rebar
    #     return curv.T * (bar_depth * mi - (sec_height / 2 - neutral_depth.T))
    
    # def get_rebar_stress(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
    #     """Computes the stress in the longitudinal rebars based on the bending direction and neutral depth.

    #     Args:
    #         neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

    #     Returns:
    #         np.ndarray: Returns the stress in the longitudinal rebars based on the bending direction and neutral depth. Returns an empty array if no rebars are defined.
    #     """
        
    #     if not self.sec.long_bar:
    #         return np.array([])
        
    #     if not isinstance(neutral_depth, np.ndarray):
    #         neutral_depth = np.array([neutral_depth])
            
    #     # Compute the strain in each rebar based on bending direction and neutral depth
    #     strain = self.get_rebar_strain(neutral_depth)
        
    #     return np.array([bar.mat.stress(eps) for bar, eps in zip(self.sec.long_bar, strain)])
    
    # def get_rebar_force(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
    #     """Computes the force in the longitudinal rebars based on the bending direction and neutral depth.

    #     Args:
    #         neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

    #     Returns:
    #         np.ndarray: Returns the force in the longitudinal rebars based on the bending direction and neutral depth. Returns an empty array if no rebars are defined.
    #     """

    #     if not self.sec.long_bar:
    #         return np.array([])

    #     if not isinstance(neutral_depth, np.ndarray):
    #         neutral_depth = np.array([neutral_depth])

    #     # Compute the stress in each rebar based on bending direction and neutral depth
    #     stress = self.get_rebar_stress(neutral_depth)

    #     return np.array([bar.area * stress for bar, stress in zip(self.sec.long_bar, stress)])

    # def get_concrete_strip_depth(self, n_disc: int = 100) -> np.ndarray:
    #     """Descretizes the concrete section and computes the depth of center of the concrete strips considering the bending direction. The strips are oriented parallel to the bending direction.

    #     Args:
    #         n_disc (int): Number of points to discretize the section. Default is 100.

    #     Returns:
    #         np.ndarray: Returns the depth of the center of the concrete strips.
    #     """
        
    #     # Compute total height of the section considering the bending direction
    #     height = self.section_height
        
    #     # Calculate the thickness of each strip based on the number of points
    #     strip_thk = height / n_disc
        
    #     return np.linspace(-height/2 + strip_thk/2, height/2 - strip_thk/2, n_disc).reshape(-1, 1)

    # def get_concrete_strip_net_area(self, n_disc: int = 100) -> np.ndarray:
    #     """Computes the net area of each concrete strip based on the bending direction and number of points. The net area is the area of the concrete strip minus the area of the rebars.

    #     Args:
    #         n_disc (int): Number of points to discretize the section. Default is 100.

    #     Returns:
    #         np.ndarray: Returns the net area of each concrete strip. Returns an array of shape (n_disc, 1) where each element is the net area of the corresponding strip. Returns an array of zeros if no rebars are defined.
    #     """
    #     if not self.sec.long_bar:
    #         return np.zeros((n_disc, 1))
        
    #     # Compute height and height of the section based on the bending direction
    #     height = self.section_height
    #     width = self.section_width
        
    #     # Compute the depth and thickness of the center of each strip based on bending direction
    #     depth = self.get_concrete_strip_depth(n_disc)
    #     thk = height / n_disc

    #     # Create concrete strips based on section orientation
    #     if self._dir in ['x', '-x']:
    #         concrete_strips = [
    #             Rectangle(width, thk, (self.sec.shape.center.x, depth[0])) for depth in depth
    #             ]
    #     else: # bending_direction in ['y', '-y']:
    #         concrete_strips = [
    #             Rectangle(thk, width, (depth[0], self.sec.shape.center.y)) for depth in depth
    #             ]

    #     # Calculate net areas of strips subtracting rebar areas
    #     bar_shapes = [bar.shape for bar in self.sec.long_bar]
    #     net_areas = np.zeros_like(concrete_strips, dtype=float)
    #     for i, strip in enumerate(concrete_strips):
    #         net_areas[i] = strip.get_net_area(bar_shapes)
            
    #     return net_areas.reshape(-1, 1)

    # def get_concrete_strain(self, neutral_depth: Union[float, np.ndarray], n_disc: int = 100) -> np.ndarray:
    #     """Computes the strain in the concrete section based on the bending direction and neutral depth.

    #     Args:
    #         neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.
    #         n_disc: Number of points to discretize the section. Default is 100.

    #     Returns:
    #         np.ndarray: Returns the strain in the concrete section based on the bending direction and neutral depth.
    #     """
    #     if not isinstance(neutral_depth, np.ndarray):
    #         neutral_depth = np.array([neutral_depth])
        
    #     # Compute curvature based on neutral depth
    #     curv = self.get_curvature(neutral_depth)

    #     # Compute total depth of the section considering the bending direction
    #     height = self.section_height

    #     # Compute the depth of the center of each strip based on bending direction
    #     depth = self.get_concrete_strip_depth(n_disc)
        
    #     # Adjust depth of concrete strips based on bending direction
    #     mi = self._depth_multiplier

    #     return curv.T * (depth * mi - (height / 2 - neutral_depth.T))

    # def get_concrete_stress(self, neutral_depth: Union[float, np.ndarray], n_disc: int = 100) -> np.ndarray:
    #     """Computes the stress in the concrete section based on the bending direction and neutral depth.

    #     Args:
    #         neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.
    #         n_disc: Number of points to discretize the section. Default is 100.

    #     Returns:
    #         np.ndarray: Returns the stress in the concrete section based on the bending direction and neutral depth.
    #     """
    #     if not isinstance(neutral_depth, np.ndarray):
    #         neutral_depth = np.array([neutral_depth])

    #     # Compute strain at the center of each strip
    #     strain = self.get_concrete_strain(neutral_depth, n_disc)
        
    #     return self.sec.mat.stress(strain)
    
    # def get_concrete_force(self, neutral_depth: Union[float, np.ndarray], n_disc: int = 100) -> np.ndarray:
    #     """Calculate forces in a rectangular concrete section due to bending.
        
    #     Args:
    #         neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction. If 'm' values are provided, it is of shape (m,) or (m, 1) and the function will compute the moment for each value.
    #         n_disc: Number of points to discretize the section. Default is 100.
            
    #     Returns:
    #         np.ndarray: Returns the force in each concrete strip based on the bending direction and neutral depth. Returns an array of shape (n_disc, 1) where each element is the force in the corresponding strip.
    #     """
    #     if not isinstance(neutral_depth, np.ndarray):
    #         neutral_depth = np.array([neutral_depth])
        
    #     # Compute the net area of each concrete strip
    #     net_areas = self.get_concrete_strip_net_area(n_disc)
        
    #     # Calculate stress at the center of each strip
    #     stress = self.get_concrete_stress(neutral_depth, n_disc)
        
    #     return stress * net_areas

    # def get_section_internal_force(self, neutral_depth: Union[float, np.ndarray], n_disc: int = 100) -> float:
    #     """Computes the total internal force in the section based on the bending direction and neutral depth. the total internal force is the sum of the forces in the longitudinal rebars (both tensile and compressive) and the concrete compression zone at the given neutral depth.

    #     Args:
    #         neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction. If 'm' values are provided, it is of shape (m,) or (m, 1) and the function will compute the moment for each value.
    #         n_disc: Number of points to discretize the section. Default is 100.

    #     Returns:
    #         float: Returns the total internal force in the section based on the bending direction and neutral depth.
    #     """
    #     if not isinstance(neutral_depth, np.ndarray):
    #         neutral_depth = np.array([neutral_depth])
        
    #     # Compute the total rebar forces at the given neutral depth
    #     rebar_forces = self.get_rebar_force(neutral_depth) # shape = (n_rebars, m)
    #     total_rebar_force = rebar_forces.sum(axis=0) # shape = (m, 1)

    #     # Compute the total concrete forces at the given neutral depth
    #     concrete_forces = self.get_concrete_force(neutral_depth, n_disc) # shape = (n_disc, m)
    #     total_concrete_force = concrete_forces.sum(axis=0) # shape = (m, 1)
        
    #     return total_rebar_force + total_concrete_force  


    # def get_section_internal_moment(self, neutral_depth: Union[float, np.ndarray], n_disc: int = 100) -> float:
    #     """Computes the total moment in the section based on the bending direction and neutral depth. The total moment is the sum of the moments in the longitudinal rebars (both tensile and compressive) and the concrete compression zone at the given neutral depth.

    #     Args:
    #         neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction. If 'm' values are provided, it is of shape (m,) or (m, 1) and the function will compute the moment for each value.
    #         n_disc: Number of points to discretize the section. Default is 100.

    #     Returns:
    #         float: Returns the total moment in the section based on the bending direction and neutral depth.
    #     """

    #     if not isinstance(neutral_depth, np.ndarray):
    #         neutral_depth = np.array([neutral_depth])
        
    #     # Compute the lever arm for the rebars and concrete strips based on the bending direction
    #     rebar_depth = self.get_long_bar_depth # shape = (n_rebars, 1)
    #     concrete_depth = self.get_concrete_strip_depth(n_disc) # shape = (n_disc, 1)
        
    #     # Compute internal forces in rebars and concrete strips
    #     rebar_forces = self.get_rebar_force(neutral_depth) # shape = (n_rebars, m)
    #     concrete_forces = self.get_concrete_force(neutral_depth, n_disc) # shape = (n_disc, m)

    #     # Compute the total moment due to rebars and concrete strips
    #     total_rebar_moment = (rebar_depth.T @ rebar_forces).sum(axis=0) # shape = (m, 1)
    #     total_concrete_moment = (concrete_depth.T @ concrete_forces).sum(axis=0) # shape = (m, 1)

    #     return total_rebar_moment + total_concrete_moment
    
    def solve_neutral_depth_equilibrium(self, external_force: float, config: SolverConfig) -> dict:
        """
        Delegates to EquilibriumSolver using SolverConfig. Returns dict with 'neutral_depth', 'force', 'moment'.
        """
        neutral_depth= self.equilibrium.solve_neutral_depth_equilibrium(external_force, config)
        internal_force = self.get_section_internal_force(neutral_depth, config.n_disc)
        internal_moment = self.get_section_internal_moment(neutral_depth, config.n_disc)
        return {
            'neutral_depth': neutral_depth,
            'force': internal_force - external_force,
            'moment': internal_moment
        }

    def get_section_internal_force(self, neutral_depth, n_disc=100):
        """Computes the total internal force in the section based on the bending direction and neutral depth. The total internal force is the sum of the forces in the longitudinal rebars (both tensile and compressive) and the concrete compression zone at the given neutral depth.
        Args:
            neutral_depth (Union[float, np.ndarray]): The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction. If 'm' values are provided, it is of shape (m,) or (m, 1) and the function will compute the moment for each value.
            n_disc (int): Number of points to discretize the section. Default is 100.
        Returns:
            np.ndarray: Returns the total internal force in the section based on the bending direction and neutral depth
        """
        rebar_force = self.force.get_rebar_force(neutral_depth) # shape = (n_rebars, m)
        concrete_force = self.force.get_concrete_force(neutral_depth, n_disc) # shape = (n_disc, m)
        return concrete_force.sum(axis=0) + rebar_force.sum(axis=0)  # shape = (m, 1)

    def get_section_internal_moment(self, neutral_depth: Union[float, np.ndarray], n_disc: int = 100) -> np.ndarray:
        """Computes the total moment in the section based on the bending direction and neutral depth. The total moment is the sum of the moments in the longitudinal rebars (both tensile and compressive) and the concrete compression zone at the given neutral depth.
        Args:
            neutral_depth (Union[float, np.ndarray]): The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction. If 'm' values are provided, it is of shape (m,) or (m, 1) and the function will compute the moment for each value.
            n_disc (int): Number of points to discretize the section. Default is 100.
        Returns:
            np.ndarray: Returns the total moment in the section based on the bending direction and neutral depth
        """
        rebar_depth = self.strain.get_long_bar_depth() # shape = (n_rebars, 1)
        rebar_force = self.force.get_rebar_force(neutral_depth) # shape = (n_rebars, m)
        concrete_depth = self.geometry.get_concrete_strip_depth(n_disc) # shape = (n_disc, 1)
        concrete_force = self.force.get_concrete_force(neutral_depth, n_disc) # shape = (n_disc, m)
        total_rebar_moment = -(rebar_depth.T @ rebar_force).sum(axis=0) if rebar_force.size and rebar_depth.size else 0 # shape = (m, 1)
        total_concrete_moment = -(concrete_depth.T @ concrete_force).sum(axis=0) if concrete_force.size and concrete_depth.size else 0 # shape = (m, 1)
        return total_rebar_moment + total_concrete_moment # shape = (m, 1)
    
    def get_neutral_depth_from_strain(self, xBar_strain: float) -> float:
        """Computes the depth of the neutral axis given the strain in the extreme tensile rebar.
        It is assumed the compressive strain the extreme compressive concrete fiber is equal to the ultimate compressive strain of the concrete material. The neutral axis depth is computed using the triangle proportionality theorem.
        Args:
            xBar_strain (float): Strain in the extreme tensile rebar.
        Returns:
            float: Returns the depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.
        """
        xfiber_coord = self.get_xFiber_coord()
        xfiber_depth = xfiber_coord.y if self._dir == BendingDirection.POSITIVE_X or self._dir == BendingDirection.NEGATIVE_X else xfiber_coord.x
        xbar_depth = self.get_xBar_depth()
        eps_cu = self.sec.mat.eps_u # negative value
        # multiplier = self.depth_multiplier()
        # return eps_cu / (eps_cu - xBar_strain) * (xfiber_depth - xbar_depth) * multiplier
        return np.abs(eps_cu / (eps_cu - xBar_strain) * (xfiber_depth - xbar_depth)) 

    def get_internal_force_moment_by_xBar_strain(self, xBar_strain: float, external_force: float, n_disc: int = 100) -> Union[dict, None]:
        """Computes the internal force and moment in the section based on the strain in the extreme tensile rebar.
        It computes the neutral depth from the strain, then calculates the internal force and moment at that depth.
        Args:
            xBar_strain (float): Strain in the extreme tensile rebar.
            external_force (float): External force applied to the section.
            n_disc (int): Number of points to discretize the section. Default is 100
        Returns:
            dict: Returns a dictionary with 'neutral_depth', 'force', and 'moment' if the section has longitudinal bars.
        Raises:
            ValueError: If the section does not have longitudinal bars.
            
        """
        if not self.sec.long_bars:
            raise ValueError("Section must have longitudinal bars to compute internal force and moment.")
        eps_s = xBar_strain
        neutral_depth = self.get_neutral_depth_from_strain(eps_s)
        internal_force = self.get_section_internal_force(neutral_depth, n_disc)
        total_force = internal_force + external_force
        total_moment = self.get_section_internal_moment(neutral_depth, n_disc)
        return {
            'neutral_depth': neutral_depth,
            'force': total_force,
            'moment': total_moment
        }
        
    def get_balance_point(self, external_force: float, n_disc: int = 100) -> Union[dict, None]:
        xbars = self.get_xBars() # list of list
        min_yield_strains = get_min_yield_strains(xbars) # positive values
        return self.get_internal_force_moment_by_xBar_strain(min_yield_strains, external_force, n_disc)

    def get_tension_controlled_point(self, external_force: float, n_disc: int = 100) -> Union[dict, None]:
        xbar_strain = 0.005
        return self.get_internal_force_moment_by_xBar_strain(xbar_strain, external_force, n_disc)
    
    def get_decompression_point(self, external_force: float, n_disc: int = 100) -> Union[dict, None]:
        xbar_strain = 0.0
        return self.get_internal_force_moment_by_xBar_strain(xbar_strain, external_force, n_disc)
        
        
    def get_pure_flexure_point(self, external_force: float, config: SolverConfig) -> dict:
        result = self.solve_neutral_depth_equilibrium(external_force, config)
        # Optionally, check equilibrium:
        # if abs(result['force']) > config.tolerance:
        #     raise ValueError("The section is not in equilibrium with the external force.")
        return result

    

        
        
    


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
    
#     if not sec.long_bar:
#         return ([], [])
    
#     bending_direction = validate_direction(bending_direction)
        
#     if bending_direction == '-x':
#         tensile_rebars = [bar for bar in sec.long_bar if bar.center.y < sec.y_max - neutral_depth]
#     elif bending_direction == 'x':
#         tensile_rebars = [bar for bar in sec.long_bar if bar.center.y > sec.y_min + neutral_depth]
#     elif bending_direction == 'y':
#         tensile_rebars = [bar for bar in sec.long_bar if bar.center.x < sec.x_max - neutral_depth]
#     elif bending_direction == '-y':
#         tensile_rebars = [bar for bar in sec.long_bar if bar.center.x > sec.x_min + neutral_depth]
    
#     compresive_rebars = [bar for bar in sec.long_bar if bar not in tensile_rebars]
    
#     return tensile_rebars, compresive_rebars


    

def nominal_tensile_strength(sec: RectangleReinforcedSection) -> dict:
    """Calculates the nominal tensile strength of the concrete section (P_nt,max) per ACI 318-25: 22.4.3.
    The nominal tensile strength is the sum of the yield strengths of all the rebars in the section.

    Args:
        sec (RectangleReinforcedSection): Concrete section containing longitudinal rebars.

    Returns:
        dict: Nominal tensile strength (P_nt,max) of the section in lbs, with keys 'neutral_depth', 'force', and 'moment'.
        'neutral_depth' is set to np.inf, 'force' is the tensile strength, and 'moment' is set to 0.0.
    """
    strength = 0.0
    for rebar in sec.long_bars:
        strength += rebar.mat.fy * rebar.area
        
    return {
        'neutral_depth': np.array([np.inf]),
        'force': np.array([strength]),
        'moment': np.array([0.0])
    }
        
        
def nominal_compressive_strength(sec: RectangleReinforcedSection) -> dict:
    """Calculates the nominal compressive strength of the concrete section (P_o) per ACI 318-25: 22.4.2..
    The nominal compressive strength is the sum of the compressive strength of the concrete and the yield strengths of all the rebars in the section.

    Args:
        sec (RectangleReinforcedSection): Concrete section containing longitudinal rebars.

    Returns:
        dict: Nominal compressive strength (P_o) of the section in lbs, with keys 'neutral_depth', 'force', and 'moment'.
        'neutral_depth' is set to -np.inf, 'force' is the compressive strength, and 'moment' is set to 0.0.
    """
    strength = 0.85 * sec.mat.fc * sec.net_area
    
    for rebar in sec.long_bars:
        strength += rebar.mat.fy * rebar.area
    
    return {
        'neutral_depth': np.array([(-np.inf)]),
        'force': np.array([-1 * strength]),
        'moment': np.array([0.0])
    }
    

def max_nominal_compressive_strength(sec: RectangleReinforcedSection) -> dict:
    """Calculates the maximum nominal compressive strength of the concrete section (P_n,max) per ACI 318-25: 22.4.2.1.
    The reduction factor is to account for accidnetal eccentricity in the section.

    Args:
        sec (RectangleReinforcedSection): Concrete section containing longitudinal rebars.

    Returns:
        dict: Maximum nominal compressive strength (P_n,max) of the section in lbs, with keys 'neutral_depth', 'force', and 'moment'.
        'neutral_depth' is set to -np.inf, 'force' is 0.80 times the compressive strength, and 'moment' is the moment at the neutral depth.
    """
    nominal = nominal_compressive_strength(sec)
    return {
        'neutral_depth': nominal['neutral_depth'],
        'force': 0.80 * nominal['force'],
        'moment': nominal['moment']
    }


def generate_force_moment_interaction(beam: UniaxialBending, min_neutral_depth: float , max_neutral_depth: float, n_points:int=100, n_disc:int=1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    neutral_depth = np.linspace(min_neutral_depth, max_neutral_depth, n_points).reshape(-1, 1)
    force = beam.get_section_internal_force(neutral_depth, n_disc)
    moment = beam.get_section_internal_moment(neutral_depth, n_disc)
    max_tension = nominal_tensile_strength(beam.sec)
    max_compression = nominal_compressive_strength(beam.sec)
    
    # Prepend max_tension and append max_compression
    neutral_depth = np.vstack((max_tension['neutral_depth'], neutral_depth, max_compression['neutral_depth']))
    force = np.hstack((max_tension['force'], force, max_compression['force']))
    moment = np.hstack((max_tension['moment'], moment, max_compression['moment']))
    
    return neutral_depth, force, moment


def get_bending_reduction_factor(beam: UniaxialBending, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
    """Calculates the strength reduction factor (Φ_b) for moment, axial force, or combined moment and axial force per ACI 318-25: Table 21.2.1.

    Args:
        beam (UniaxialBending): Concrete section containing longitudinal and transverse rebars.
        neutral_depth (Union[float, np.ndarray]): Depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction. Size of the neutral depth could be (m,) or (m, 1).

    Returns:
        np.ndarray: Strength reduction factor (Φ_b) of the section for all the provided neutral depth. 
    """
    neutral_depth = np.asarray(neutral_depth, dtype=float).reshape(-1)
    
    # Determine whether spiral transverse reinforcement is present
    trans_bars = beam.sec.trans_bars
    spiral_trans_bars = [bar for bar in trans_bars if bar.is_looped and bar.is_spiral]
    has_spiral = len(spiral_trans_bars) != 0
    phi_compression_axial = 0.75 if has_spiral else 0.65
    
    # Initialize phi array
    phi = np.empty_like(neutral_depth)
    
    # Handle np.inf cases
    is_inf_mask = np.isinf(neutral_depth) # both positive and negative infinity
    is_posinf_mask = np.isposinf(neutral_depth)
    is_neginf_mask = np.isneginf(neutral_depth)
    
    # For entries with neutral_depth == inf
    phi[is_posinf_mask] = 0.90
    phi[is_neginf_mask] = phi_compression_axial
    
    # For all other (finite) neutral depths
    if np.any(~is_inf_mask):
        finite_nd = neutral_depth[~is_inf_mask]
        xbars = beam.get_xBars(finite_nd)
        xbar_strains = beam.get_rebar_strain(finite_nd)
        min_yield_strains = get_min_yield_strains(xbars)
        max_xbar_strain_ratio = np.max(xbar_strains / min_yield_strains, axis=0)

        is_nan = np.isnan(max_xbar_strain_ratio)
        is_compression_controlled = max_xbar_strain_ratio <= 1
        is_tension_controlled = max_xbar_strain_ratio >= 1 + 0.003 / min_yield_strains

        if has_spiral:
            phi_finite = np.where(is_nan, 0.75,
                          np.where(is_compression_controlled, 0.75,
                          np.where(is_tension_controlled, 0.90,
                              0.75 + 0.15 * (max_xbar_strain_ratio - 1) / (0.003 / min_yield_strains))))
        else:
            phi_finite = np.where(is_nan, 0.65,
                          np.where(is_compression_controlled, 0.65,
                          np.where(is_tension_controlled, 0.90,
                              0.65 + 0.25 * (max_xbar_strain_ratio - 1) / (0.003 / min_yield_strains))))
        
        phi[~is_inf_mask] = phi_finite

    return phi



# Utility Functions =========================================================
def get_min_yield_strains(bar_groups: List[List[LongitudinalRebar]]) -> np.ndarray:
    """Returns the minimum yield strain of the rebars in each group.
    Args:
        bar_groups (List[List[LongitudinalRebar]]): List of lists containing longitudinal rebars.
    Returns:
        np.ndarray: Returns an array of minimum yield strains for each group. If a group has no bars, it returns NaN for that group.
    """
    return np.array([
        min(bar.mat.eps_y for bar in bars) if len(bars) != 0 else np.nan
        for bars in bar_groups
    ])


# Main function =========================================================      
def main():
    pass
    
    
if __name__ == "__main__":
    main()