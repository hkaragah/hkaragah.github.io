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

from assets.modules.shapes import Point, Rectangle, Circle
from assets.modules.shapes import shapes
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


class UniaixalBending:
    """A class representing uniaxial bending of a rectangular concrete section.
    This class is used to compute the bending capacity of a concrete section under uniaxial bending.
    """
    def __init__(self, sec: RectangleReinforcedSection, bending_direction: str):
        """Initializes the UniaxialBending object.

        Args:
            con (RectangleReinforcedSection): Concrete section object containing the longitudinal rebars.
            bending_direction (str): Direction of the bending, must be one of:
                'x': when bending vector is in the positive x direction (bending about the x-axis causing compression on the positive y side)
                '-x': when bending vector is in the negative x direction (bending about the x-axis causing compression on the negative y side)
                'y': when bending vector is in the positive y direction (bending about the y-axis causing compression on the positive x side)
                '-y': when bending vector is in the negative y direction (bending about the y-axis causing compression on the negative x side)

        Raises:
            
        """
        
        if not isinstance(sec, RectangleReinforcedSection):
            raise TypeError("sec must be an instance of RectangleReinforcedSection.")
        
        self._sec = sec
        self._dir = self._validate_direction(bending_direction)
        
    @property
    def sec(self) -> RectangleReinforcedSection:
        """Returns the concrete section."""
        return self._sec
    
    @property
    def dir(self) -> str:
        """Returns the bending direction."""
        return self._dir
    
    @property
    def _depth_multiplier(self) -> int:
        """Returns the multiplier for the depth depending on the bending direction.

        Returns:
            int: 1 if the '-x' and '+y' bending direction, -1 if the bending direction is '+x' or '-y'.
        """
        return 1 if self._dir in ['-x', 'y'] else -1
    
    @staticmethod
    def _validate_direction(direction: str) -> str:
        """Checks if the given direction is valid.
        
        Args:
            direction (str): Direction string to check.
        
        Raises:
            TypeError: If the direction is not a string.
            ValueError: If the direction is None or not one of the valid options.
        """
        valid_directions = ['x', '-x', 'y', '-y']
        
        if not isinstance(direction, str):
            raise TypeError("Direction must be a string.")
        
        if direction is None:
            raise ValueError("Direction cannot be None.")
        
        direction = direction.lower()
        if direction not in valid_directions:
            raise ValueError(f"Direction must be one of {valid_directions}.")
        
        return direction
    
    @property
    def net_area(self) -> float:
        """Returns the net area of the section."""
        lBar_area = sum([bar.area for bar in self.sec.lBars])
        return self.sec.gross_area - lBar_area

    @property
    def tBar_area(self) -> float:
        """Returns the total area of transverse bars in the section."""
        if not self.sec.tBars:
            return 0.0
        
        if self._dir in ['x', '-x']:
            return sum(bar.Ax for bar in self.sec.tBars)
        elif self._dir in ['y', '-y']:
            return sum(bar.Ay for bar in self.sec.tBars) 

    def get_tensile_rebars(self, neutral_depth: Union[float, np.ndarray]) -> List[LongitudinalRebar]:
        """Returns the list of tensile rebars in the section based on the bending direction and neutral depth. The tensile rebars are those that are located in the tension zone of the section and are not marked as skin rebars.

        Args:
            neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

        Returns:
            List[LongitudinalRebar]: Returns the list of tensile rebars in the section based on the bending direction and neutral depth. Returns an empty list if no rebars are defined.
        """
        if not self.sec.lBars:
            return []
        
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        # Define condition functions for each direction
        conditions:dict[str, Callable[[LongitudinalRebar], bool]] = {
            '-x': lambda bar: bar.center.y < self.sec.shape.y_max - neutral_depth,
            'x': lambda bar: bar.center.y > self.sec.shape.y_min + neutral_depth,
            'y': lambda bar: bar.center.x < self.sec.shape.x_max - neutral_depth,
            '-y': lambda bar: bar.center.x > self.sec.shape.x_min + neutral_depth
        }
        
        return [
            bar for bar in self.sec.lBars if conditions[self._dir](bar) and not bar.is_skin_bar
        ]

    def get_compressive_rebars(self, neutral_depth: Union[float, np.ndarray]) -> List[LongitudinalRebar]:
        """Returns the list of compressive rebars in the section based on the bending direction and neutral depth. The compressive rebars are those that are located in the compression zone of the section and are not marked as skin rebars.

        Args:
            neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

        Returns:
            List[LongitudinalRebar]: Returns the list of compressive rebars in the section based on the bending direction and neutral depth. Returns an empty list if no rebars are defined.
        """
        
        if not self.sec.lBars:
            return []
        
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        tensile_rebars = self.get_tensile_rebars(neutral_depth)
        
        return [bar for bar in self.sec.lBars if bar not in tensile_rebars and not bar.is_skin_bar]
    
    @property
    def get_xFiber_coord(self) -> Point:
        """Returns the coordinates of the extreme compressive fiber for the bending direction in a rectangular concrete section. The point of extreme compressive fiber is the location of the fiber of maximum compressive strain.

        Returns:
            Point: Point of extreme compressive fiber considering the bending direction.
        """
        ref_points = {
            'x' : Point(self.sec.shape.center.x, self.sec.shape.y_min),
            '-x': Point(self.sec.shape.center.x, self.sec.shape.y_max),
            'y' : Point(self.sec.shape.x_max, self.sec.shape.center.y),
            '-y': Point(self.sec.shape.x_min, self.sec.shape.center.y)
        }
        
        return ref_points.get(self._dir, None)
    
    def get_tensile_effective_depth(self, neutral_depth: Union[float, np.ndarray]):
        """Computes the distance from extreme compression fiber to centroid of longitudinal tension reinforcement. If no rebar is defined, it returns None.
        
        Args:
            neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

        Returns:
            float: Distance from extreme compression fiber to centroid of tension reinforcement.
            None: If no rebars are defined.

        """
        
        if not self.sec.lBars:
            return None
        
        # Compute the coornidates of the extreme compressive fiber based on the bending direction
        xFiber_coord = self.get_xFiber_coord
        
        # Compute the centroid of the longitudinal tension reinforcement based on the bending direction
        tensile_rebars = self.get_tensile_rebars(neutral_depth)
        bar_shapes = [bar.shape for bar in tensile_rebars]
        bar_centroid = shapes.get_circle_group_centroid(bar_shapes)
        
        return bar_centroid.distance_to(xFiber_coord)

    def get_compressive_effective_depth(self, neutral_depth: Union[float, np.ndarray]):
        """Computes the distance from extreme compression fiber to centroid of longitudinal compression reinforcement. If no rebar is defined, it returns None.
        
        Args:
            neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

        Returns:
            float: Distance from extreme compression fiber to centroid of tension reinforcement.
            None: If no rebars are defined.
        """
        
        if not self.sec.lBars:
            return None
        
        if not isinstance(neutral_depth, np.ndarray):   
            neutral_depth = np.array([neutral_depth])

        # Compute the coornidates of the extreme compressive fiber based on the bending direction
        xFiber_coord = self.get_xFiber_coord
        
        # Compute the centroid of the longitudinal compression reinforcement based on the bending direction
        compressive_rebars = self.get_compressive_rebars(neutral_depth)
        bar_shapes = [bar.shape for bar in compressive_rebars]
        bar_centroid = shapes.get_circle_group_centroid(bar_shapes)
        
        return bar_centroid.distance_to(xFiber_coord)
            
        # if bending_direction == 'x':
        #     return abs(self.x_max - self.lBars_x_max)
        # elif bending_direction == '-x':
        #     return abs(self.x_min - self.lBars_x_min)
        # elif bending_direction == 'y':
        #     return abs(self.y_max - self.lBars_y_max)
        # elif bending_direction == '-y':
        #     return abs(self.y_min - self.lBars_y_min)        
     
    @property
    def get_xBar_depth(self):
        """Returns the depth of the extreme outer most rebars in the section based on the bending direction. The depth of th outer most tensile rebar is measured from the section centroid to the centroid of the outer most tensile rebar. Returns None if no rebars are defined.
        """
        if not self.sec.lBars:
            return None
        
        min_depth = {
            'x' : max([bar.center.y for bar in self.sec.lBars]),
            '-x': min([bar.center.y for bar in self.sec.lBars]),
            'y' : min([bar.center.x for bar in self.sec.lBars]),
            '-y': max([bar.center.x for bar in self.sec.lBars])
        }
        return min_depth[self._dir]
                
    @property
    def get_xBars(self) -> List[LongitudinalRebar]:
        """Return the list of extreme outer most tensile rebars in the section based on the bending direction. The outer most tensile rebar is the one that is located at the outer most depth from the extreme compressive fiber. Returns an empty list if no rebars are defined.
        """
        
        if not self.sec.lBars:
            return []

        # Find the depth of the outer most tensile rebar based on the bending direction
        depth = self.get_xBar_depth
        
        # Find the rebars at the outer most depth based on the bending direction
        if self._dir in ['x', '-x']:
            return [bar for bar in self.sec.lBars if bar.center.y == depth]
        else:
            return [bar for bar in self.sec.lBars if bar.center.x == depth]        

    @property
    def get_lBar_depth(self) -> np.ndarray:
        """Returns the depth of each rebar from the extreme compressive fiber based on the bending direction.

        Returns:
            np.ndarray: Returns the depth of each rebar from the extreme compressive fiber based on the bending direction. Returns an empty array if no rebars are defined.
        """
        
        if not self.sec.lBars:
            return np.array([])
        
        # Compute the depth of each rebar based on bending direction
        if self._dir in ['x', '-x']:
            return np.array([bar.center.y for bar in self.sec.lBars]).reshape(-1, 1)
        else: # bending_direction in ['y', '-y']
            return np.array([bar.center.x for bar in self.sec.lBars]).reshape(-1, 1)  
        
    @property
    def section_height(self) -> float:
        """Returns the height of the section based on the bending direction."""
        if self._dir in ['x', '-x']:
            return self.sec.shape.y_max - self.sec.shape.y_min
        else: # bending_direction in ['y', '-y']
            return self.sec.shape.x_max - self.sec.shape.x_min
        
    @property
    def section_width(self) -> float:
        """Returns the width of the section based on the bending direction."""
        if self._dir in ['x', '-x']:
            return self.sec.shape.x_max - self.sec.shape.x_min
        else: # bending_direction in ['y', '-y']
            return self.sec.shape.y_max - self.sec.shape.y_min

    def get_curvature(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the curvature of the section based on the bending direction and neutral depth.
        The curvature is computed as the ratio of the ultimate compressive strain in concrete to the depth of the neutral axis.

        Args:
            neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

        Returns:
            np.ndarray: Returns the curvature of the section based on the bending direction and neutral depth.
            
        Raises:
            ValueError: If neutral_depth is zero.
        """
        
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        try:
            return np.divide(self.sec.mat.eps_u, neutral_depth)
        except ZeroDivisionError:
            raise ValueError("Neutral depth cannot be zero.")

    def get_rebar_strain(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
        """Computes the strain in the longitudinal rebars based on the bending direction and neutral depth.

        Args:
            neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

        Returns:
            np.ndarray: Returns the strain in the longitudinal rebars based on the bending direction and neutral depth. Returns an empty array if no rebars are defined.
        """
        if not self.sec.lBars:
            return np.array([])
        
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        # Compute curvature based on neutral depth
        curv = self.get_curvature(neutral_depth)

        # Compute total depth of the section considering the bending direction
        sec_height = self.section_height
        
        # Compute the depth of each rebar relative to the section centroid based on bending direction
        bar_depth = self.get_lBar_depth 

        # Adjust depth based on bending direction
        mi = self._depth_multiplier

        # Calculate strain, stress, and force for each rebar
        return curv.T * (bar_depth * mi - (sec_height / 2 - neutral_depth.T))
    
    def get_rebar_stress(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
        """Computes the stress in the longitudinal rebars based on the bending direction and neutral depth.

        Args:
            neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

        Returns:
            np.ndarray: Returns the stress in the longitudinal rebars based on the bending direction and neutral depth. Returns an empty array if no rebars are defined.
        """
        
        if not self.sec.lBars:
            return np.array([])
        
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
            
        # Compute the strain in each rebar based on bending direction and neutral depth
        strain = self.get_rebar_strain(neutral_depth)
        
        return np.array([bar.mat.stress(eps) for bar, eps in zip(self.sec.lBars, strain)])
    
    def get_rebar_force(self, neutral_depth: Union[float, np.ndarray]) -> np.ndarray:
        """Computes the force in the longitudinal rebars based on the bending direction and neutral depth.

        Args:
            neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

        Returns:
            np.ndarray: Returns the force in the longitudinal rebars based on the bending direction and neutral depth. Returns an empty array if no rebars are defined.
        """

        if not self.sec.lBars:
            return np.array([])

        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])

        # Compute the stress in each rebar based on bending direction and neutral depth
        stress = self.get_rebar_stress(neutral_depth)

        return np.array([bar.area * stress for bar, stress in zip(self.sec.lBars, stress)])

    def get_concrete_strip_depth(self, n_points: int = 100) -> np.ndarray:
        """Descretizes the concrete section and computes the depth of center of the concrete strips considering the bending direction. The strips are oriented parallel to the bending direction.

        Args:
            n_points (int): Number of points to discretize the section. Default is 100.

        Returns:
            np.ndarray: Returns the depth of the center of the concrete strips.
        """
        
        # Compute total height of the section considering the bending direction
        height = self.section_height
        
        # Calculate the thickness of each strip based on the number of points
        strip_thk = height / n_points
        
        return np.linspace(-height/2 + strip_thk/2, height/2 - strip_thk/2, n_points).reshape(-1, 1)

    def get_concrete_strip_net_area(self, n_points: int = 100) -> np.ndarray:
        """Computes the net area of each concrete strip based on the bending direction and number of points. The net area is the area of the concrete strip minus the area of the rebars.

        Args:
            n_points (int): Number of points to discretize the section. Default is 100.

        Returns:
            np.ndarray: Returns the net area of each concrete strip. Returns an array of shape (n_points, 1) where each element is the net area of the corresponding strip. Returns an array of zeros if no rebars are defined.
        """
        if not self.sec.lBars:
            return np.zeros((n_points, 1))
        
        # Compute height and height of the section based on the bending direction
        height = self.section_height
        width = self.section_width
        
        # Compute the depth and thickness of the center of each strip based on bending direction
        depth = self.get_concrete_strip_depth(n_points)
        thk = height / n_points

        # Create concrete strips based on section orientation
        if self._dir in ['x', '-x']:
            concrete_strips = [
                Rectangle(width, thk, (self.sec.shape.center.x, depth[0])) for depth in depth
                ]
        else: # bending_direction in ['y', '-y']:
            concrete_strips = [
                Rectangle(thk, width, (depth[0], self.sec.shape.center.y)) for depth in depth
                ]

        # Calculate net areas of strips subtracting rebar areas
        bar_shapes = [bar.shape for bar in self.sec.lBars]
        net_areas = np.zeros_like(concrete_strips, dtype=float)
        for i, strip in enumerate(concrete_strips):
            net_areas[i] = strip.get_net_area(bar_shapes)
            
        return net_areas.reshape(-1, 1)

    def get_concrete_strain(self, neutral_depth: Union[float, np.ndarray], n_points: int = 100) -> np.ndarray:
        """Computes the strain in the concrete section based on the bending direction and neutral depth.

        Args:
            neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.
            n_points: Number of points to discretize the section. Default is 100.

        Returns:
            np.ndarray: Returns the strain in the concrete section based on the bending direction and neutral depth.
        """
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        # Compute curvature based on neutral depth
        curv = self.get_curvature(neutral_depth)

        # Compute total depth of the section considering the bending direction
        height = self.section_height

        # Compute the depth of the center of each strip based on bending direction
        depth = self.get_concrete_strip_depth(n_points)
        
        # Adjust depth of concrete strips based on bending direction
        mi = self._depth_multiplier

        return curv.T * (depth * mi - (height / 2 - neutral_depth.T))

    def get_concrete_stress(self, neutral_depth: Union[float, np.ndarray], n_points: int = 100) -> np.ndarray:
        """Computes the stress in the concrete section based on the bending direction and neutral depth.

        Args:
            neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.
            n_points: Number of points to discretize the section. Default is 100.

        Returns:
            np.ndarray: Returns the stress in the concrete section based on the bending direction and neutral depth.
        """
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])

        # Compute strain at the center of each strip
        strain = self.get_concrete_strain(neutral_depth, n_points)
        
        return self.sec.mat.stress(strain)
    
    def get_concrete_force(self, neutral_depth: Union[float, np.ndarray], n_points: int = 100) -> np.ndarray:
        """Calculate forces in a rectangular concrete section due to bending.
        
        Args:
            neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction. If 'm' values are provided, it is of shape (m,) or (m, 1) and the function will compute the moment for each value.
            n_points: Number of points to discretize the section. Default is 100.
            
        Returns:
            np.ndarray: Returns the force in each concrete strip based on the bending direction and neutral depth. Returns an array of shape (n_points, 1) where each element is the force in the corresponding strip.
        """
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        # Compute the net area of each concrete strip
        net_areas = self.get_concrete_strip_net_area(n_points)
        
        # Calculate stress at the center of each strip
        stress = self.get_concrete_stress(neutral_depth, n_points)
        
        return stress * net_areas

    def get_section_internal_force(self, neutral_depth: Union[float, np.ndarray], n_points: int = 100) -> float:
        """Computes the total internal force in the section based on the bending direction and neutral depth. the total internal force is the sum of the forces in the longitudinal rebars (both tensile and compressive) and the concrete compression zone at the given neutral depth.

        Args:
            neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction. If 'm' values are provided, it is of shape (m,) or (m, 1) and the function will compute the moment for each value.
            n_points: Number of points to discretize the section. Default is 100.

        Returns:
            float: Returns the total internal force in the section based on the bending direction and neutral depth.
        """
        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        # Compute the total rebar forces at the given neutral depth
        rebar_forces = self.get_rebar_force(neutral_depth) # shape = (n_rebars, m)
        total_rebar_force = rebar_forces.sum(axis=0) # shape = (m, 1)

        # Compute the total concrete forces at the given neutral depth
        concrete_forces = self.get_concrete_force(neutral_depth, n_points) # shape = (n_points, m)
        total_concrete_force = concrete_forces.sum(axis=0) # shape = (m, 1)
        
        return total_rebar_force + total_concrete_force  


    def get_section_internal_moment(self, neutral_depth: Union[float, np.ndarray], n_points: int = 100) -> float:
        """Computes the total moment in the section based on the bending direction and neutral depth. The total moment is the sum of the moments in the longitudinal rebars (both tensile and compressive) and the concrete compression zone at the given neutral depth.

        Args:
            neutral_depth Union[float, np.ndarray]: The depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction. If 'm' values are provided, it is of shape (m,) or (m, 1) and the function will compute the moment for each value.
            n_points: Number of points to discretize the section. Default is 100.

        Returns:
            float: Returns the total moment in the section based on the bending direction and neutral depth.
        """

        if not isinstance(neutral_depth, np.ndarray):
            neutral_depth = np.array([neutral_depth])
        
        # Compute the lever arm for the rebars and concrete strips based on the bending direction
        rebar_depth = self.get_lBar_depth # shape = (n_rebars, 1)
        concrete_depth = self.get_concrete_strip_depth(n_points) # shape = (n_points, 1)
        
        # Compute internal forces in rebars and concrete strips
        rebar_forces = self.get_rebar_force(neutral_depth) # shape = (n_rebars, m)
        concrete_forces = self.get_concrete_force(neutral_depth, n_points) # shape = (n_points, m)

        # Compute the total moment due to rebars and concrete strips
        total_rebar_moment = (rebar_depth.T @ rebar_forces).sum(axis=0) # shape = (m, 1)
        total_concrete_moment = (concrete_depth.T @ concrete_forces).sum(axis=0) # shape = (m, 1)

        return total_rebar_moment + total_concrete_moment
    
    def solve_neutral_depth_equilibrium(self, external_force:float, method:str, n_points: int = 100, **kwargs) -> Tuple[float, float]: 
        """Computes the depth of the neutral axis based on the external force applied to the section.
        The neutral axis is the depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.

        Args:
            external_force (float): External force applied to the section. Use positive values for tension and negative values for compression.
            method (str): Method to be used for computing the neutral axis. Must be one of:
                'bisect': Bisection method
                'brentq': Brent's method
                'secant': Secant method
                'newton': Newton's method
            n_points (int): Number of points to discretize the section. Default is 100.
            **kwargs: Additional keyword arguments for the root-finding method, such as:
                tol (float): Tolerance for the root-finding method. Default is 1e-2.
                max_iter (int): Maximum number of iterations for the root-finding method. Default is 100.
                x0 (float): Initial guess for the root-finding method. Default is 1e-2.
                x1 (float): Upper bound for the root-finding method. Default is section height - 1e-2.
                
        Returns:
            float: Returns the depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.
            float: Returns the tolerance of the root-finding method.
            
        Raises:
            ValueError: If the method is not one of the valid options or if the root-finding method fails to converge.
        """
                
        tol = kwargs.get('tol', 1e-2)
        max_iter = kwargs.get('max_iter', 100)
        height = self.section_height
        x0 = kwargs.get('x0', 1e-1)
        x1 = kwargs.get('x1', height - 1e-1)
        
        def equilibrium(neutral_depth: float) -> float:    
            internal_force = self.get_section_internal_force(neutral_depth, n_points)
            return internal_force + external_force

        if method in ['bisect', 'brentq']:
            result = root_scalar(equilibrium, bracket=[x0, x1], method=method, xtol=tol, maxiter=max_iter)
            
        elif method in ['secant', 'newton']:
            result = root_scalar(equilibrium, method=method, x0=x0, x1=x1, xtol=tol, maxiter=max_iter)
            
        if result.converged:
            return {
                'neutral_depth': result.root, 
                'tol': tol
                }
        else:
            raise ValueError("Failed to find a solution for the neutral axis depth.")    
    
    def get_neutral_depthfrom_strain(self, xBar_strain: float) -> float:
        """Computes the depth of the neutral axis given the strain in the extreme tensile rebar.
        It is assumed the compressive strain the extreme compressive concrete fiber is equal to the ultimate compressive strain of the concrete material. The neutral axis depth is computed using the triangle proportionality theorem.
        
        Args:
            xBar_strain (float): Strain in the extreme tensile rebar.

        Returns:
            float: Returns the depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.
        """
        # Find the depth of the extreme compressive concrete fiber
        xfiber_coord = self.get_xFiber_coord
        xfiber_depth = xfiber_coord.y if self._dir in ['x', '-x'] else xfiber_coord.x
        
        # Find depth of the extreme tensile rebars
        xbar_depth = self.get_xBar_depth
        
        # find concrete ultimate compressive strain (a negative value)
        eps_cu = self.sec.mat.eps_u
        
        # Compute the neutral axis depth
        multiplier = self._depth_multiplier
        
        return eps_cu / (eps_cu - xBar_strain) * (xfiber_depth - xbar_depth) * multiplier
    
    def get_balance_point(self, external_force:float):
        
        if not self.sec.lBars:
            return None

        # Find extreme tensile rebars
        xbars = self.get_xBars

        # Find minimum yield strain among the extreme tensile rebars (a positive value)
        eps_y_min = max([bar.mat.eps_y for bar in xbars])
        
        # Compute the neutral axis depth
        neutral_depth = self.get_neutral_depthfrom_strain(eps_y_min,)

        # Compute total internal forces
        internal_force = self.get_section_internal_force(neutral_depth)
        total_force = internal_force + external_force
        
        # Compute total internal moments
        total_moment = self.get_section_internal_moment(neutral_depth)
        
        return {
            'neutral_depth': neutral_depth,
            'force': total_force,
            'moment': total_moment
        }

    def get_tension_conrolled_point(self, external_force:float):
        if not self.sec.lBars:
            return None
        
        # Strain in the extreme tensile rebar
        eps_s = 0.005
        
        # Compute the neutral axis depth
        neutral_depth = self.get_neutral_depthfrom_strain(eps_s)
        
        # Compute total internal forces
        internal_force = self.get_section_internal_force(neutral_depth)
        total_force = internal_force + external_force
        
        # Compute total internal moments
        total_moment = self.get_section_internal_moment(neutral_depth)
        
        return {
            'neutral_depth': neutral_depth,
            'force': total_force,
            'moment': total_moment
        }
    
    def get_pure_flexure_point(self,external_force:float, method: str= 'secant', **kwargs) -> dict:
        
        # Solve for the neutral depth at equilibrium with the external force
        neutral_depth, tol = self.solve_neutral_depth_equilibrium(external_force, method, **kwargs)
        
        # Compute total forces in the section
        internal_force = self.get_section_internal_force(neutral_depth)
        total_force = internal_force + external_force
        
        # Check if the section is in equilibrium with the external force
        if abs(total_force) > tol:
            raise ValueError("The section is not in equilibrium with the external force.")
    
        # Compute total moments in the section
        total_moment = self.get_section_internal_moment(neutral_depth)
        
        return {
            'neutral_depth': neutral_depth,
            'force': total_force,
            'moment': total_moment
        }
    
@dataclass
class SolverConfig:
    method: str = 'secant'
    tolerance: float = 1e-2
    max_iterations: int = 100
    n_discretization_points: int = 100
        
        
    


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