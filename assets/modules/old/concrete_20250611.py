################################
# Concrete Section Module
# Written by: Hossein Karagah
# Date: 2025-03-10
# Description: This module provides classes and functions to model and analyze rectangular concrete sections with rebars.
################################


from copy import deepcopy
from assets.modules.shapes import Point
import assets.modules.shapes as sh
from assets.modules.materials import ACIConcrete, Concrete
from assets.modules.steel import RebarSection, TransverseRebar, SpiralRebar
from typing import Tuple, List, Union, Optional
import numpy as np
import matplotlib.axes as axs
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, fsolve

##############################################
# Global variables
##############################################



##############################################
class RectangleConcreteSection:
    def __init__(self, shape: sh.Rectangle, concrete_mat: Concrete, cover: float, center: Union[Point, Tuple[float, float]] = (0.0, 0.0)):
        self._shape = deepcopy(shape)
        self._shape.center = center
        self.mat = concrete_mat
        if shape.width - 2 * cover <= 0 or shape.height - 2 * cover <= 0:
            raise ValueError("Cover is too large relative to the section dimensions.")
        self.cover = cover
        self.rebars: List[RebarSection] = []
        self.tBars: List[(TransverseRebar, bool)] = []
        
    @property
    def shape(self) -> sh.Rectangle:
        return self._shape
    
    @property
    def corners(self) -> List[Point]:
        """Returns the corners of the rectangle section."""
        return self._shape.corners
    
    @property
    def gross_area(self) -> float:
        return self._shape.area
    
    @property
    def net_area(self) -> float:
        rebar_area = sum([rebar.area for rebar in self.rebars])
        return self.gross_area - rebar_area
    
    @property
    def center(self) -> Point:
        return Point(self._shape.center.x, self._shape.center.y)
    
    @center.setter
    def center(self, new_center: Union[Point, Tuple[float, float]]):
        if isinstance(new_center, tuple):
            new_center = Point(new_center[0], new_center[1])
        self._shape.center = new_center
        
    def translate(self, dx: float, dy: float):
        """Translates the section by dx and dy."""
        self._shape.translate(dx, dy)
        for rebar in self.rebars:
            rebar.translate(dx, dy)
    
    def rotate(self, angle_deg: float, pivot: Union[Point, Tuple[float, float]] = None):
        """Rotates the section by angle_deg around the pivot point."""
        self._shape.rotate(angle_deg, pivot)
        for rebar in self.rebars:
            if pivot is None:
                rebar_pivot = self._shape.center
            else:
                rebar_pivot = pivot
            rebar.rotate(angle_deg, rebar_pivot)       
            
    def core(self, rebar: Optional[RebarSection] = None) -> sh.Rectangle:
        """Computes the concrete core excluding the cover, transverse bars, and optionally a rebar.
        The core is the area where the main rebars can be placed without overlapping with the cover or transverse bars.

        Args:
            rebar (Optional[RebarSection], optional): longitudinal rabar. Defaults to None.

        Returns:
            sh.Rectangle: The core rectangle shape that defines the area where longitudinal rebars can be placed.
        """
        if rebar is None:
            offset = 0
        else:
            offset = rebar.dia
            
        width = self._shape.width - 2 * self.cover - 2 * self.max_tBar_dia - offset
        height = self._shape.height - 2 * self.cover - 2 * self.max_tBar_dia - offset
        
        return sh.Rectangle(width, height, self._shape.center, self._shape.rotation)
    
    @property
    def max_tBar_dia(self) -> float:
        if not self.tBars:
            return 0.0
        else:
            return max(tBar.rebar.dia if isOuter else 0 
                          for tBar, isOuter in self.tBars)
    
    def add_tBar(self, tBar: TransverseRebar, isOuter: bool = True):
        self.tBars.append((deepcopy(tBar), isOuter))
        
    def add_rebar(self, rebar: RebarSection, center: Union[Point, Tuple[float, float]]):
        """Adds a rebar to the wall.
        The rebar is added at the specified center point. The function checks if the new rebar overlaps with existing rebars and if it is within the wall bounds minus cover.

        Args:
            rebar (RebarSection): RebarSection object to be added.
            center (Union[Point, Tuple[float, float]]): Center point of the rebar.
                If a tuple is provided, it will be converted to a Point object.

        Raises:
            ValueError: If the new rebar overlaps with existing rebars or if it is outside the wall bounds minus cover.
            ValueError: If the rebar is outside the wall bounds minus cover.
        """
        if isinstance(center, tuple):
            center = Point(center[0], center[1])
        
        bar = deepcopy(rebar)
        bar.center = center
        
        # Check if new rebar overlaps with existing rebars
        for existing_rebar in self.rebars:
            if sh.circles_overlap(bar.shape, existing_rebar.shape):
                raise ValueError(f"Rebar at {bar.center} overlaps with existing rebar at {existing_rebar.center}.")
        
        # Check if new rebar is within core region
        if not sh.point_within_rectangle(bar.center, self.core(bar)):
            core =self.core(bar)
            corners = core.corners
            x_min, y_min, x_max, y_max = corners[0].x, corners[0].y, corners[2].x, corners[2].y
            raise ValueError(f"Rebar at {bar.center} is outside the core region ({x_min, y_min}), ({x_max, y_max}).")
        
        self.rebars.append(bar)

    def add_rebar_row(self, rebar: RebarSection, center: Union[Point, Tuple[float, float]], n: int, direction: Tuple[float, float], spacing: Optional[float]=None):
        """Adds a row of rebars to the wall section.
        
        Args:
            rebar (RebarSection): Rebar section to add.
            center (Union[Point, Tuple[float, float]]): Center point of the row.
            n (int): Number of rebars in the row.
            direction (Tuple[float, float]): Unit vector specifying direction of the row.
            spacing (float): Spacing between rebars.
            
        Raises:
            ValueError: If rebars would overlap with cover region or transverse bars.
        """
        
        # Convert tuple center to Point if needed
        if isinstance(center, tuple):
            center = Point(center[0], center[1])
            
        # Normalize direction vector
        direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Direction vector cannot be zero.")
        dx, dy = direction / norm
        
        # Use core region for placement
        core_rect = self.core(rebar)
        corners = core_rect.corners
        x_vals = [pt.x for pt in corners]
        y_vals = [pt.y for pt in corners]
        min_x = min(x_vals)
        max_x = max(x_vals)
        min_y = min(y_vals)
        max_y = max(y_vals)

        if spacing is None:
            if dx != 0:
                max_dist_x = min(center.x - min_x, max_x - center.x) / abs(dx)
            else:
                max_dist_x = float('inf')

            if dy != 0:
                max_dist_y = min(center.y - min_y, max_y - center.y) / abs(dy)
            else:
                max_dist_y = float('inf')

            max_dist = min(max_dist_x, max_dist_y)
            spacing = max_dist / ((n - 1) / 2)

        if spacing <= 0:
            raise ValueError("Invalid spacing: too small or negative.")

        # Add rebars
        for i in range((n-1)//2 + 1):
            if n%2 == 0:
                offset = (2*i + 1) * spacing / 2
            else:
                offset = i * spacing

            if n%2 == 1 and i == 0:
                pos = Point(center.x, center.y)
                self.add_rebar(rebar, pos)
            else:
                pos1 = Point(center.x + dx * offset, center.y + dy * offset)
                pos2 = Point(center.x - dx * offset, center.y - dy * offset)
                self.add_rebar(rebar, pos1)
                self.add_rebar(rebar, pos2)
    
    @property
    def rebars_max_x(self):
        """Returns the maximum x coordinate of all rebars in the section."""
        if not self.rebars:
            return None
        return max([rebar.center.x for rebar in self.rebars])
    
    @property
    def rebars_min_x(self):
        """Returns the minimum x coordinate of all rebars in the section."""
        if not self.rebars:
            return None
        return min([rebar.center.x for rebar in self.rebars])
    
    @property
    def rebars_max_y(self):
        """Returns the maximum y coordinate of all rebars in the section."""
        if not self.rebars:
            return None
        return max([rebar.center.y for rebar in self.rebars])
    
    @property
    def rebars_min_y(self):
        """Returns the minimum y coordinate of all rebars in the section."""
        if not self.rebars:
            return None
        return min([rebar.center.y for rebar in self.rebars])
           
    @property
    def section_min_x(self):
        """Returns the minimum x coordinate of the section."""
        return min([corner.x for corner in self.corners])
    
    @property
    def section_max_x(self):
        """Returns the maximum x coordinate of the section."""
        return max([corner.x for corner in self.corners])
    
    @property
    def section_min_y(self):
        """Returns the minimum y coordinate of the section."""
        return min([corner.y for corner in self.corners])
    
    @property
    def section_max_y(self):
        """Returns the maximum y coordinate of the section."""
        return max([corner.y for corner in self.corners])
            
    def tensile_effective_depth(self, bending_direction: str):
        """Computes the distance from extreme compression fiber to 
        centroid of longitudinal tension reinforcement.
        If no rebar is defined, it returns None.
        
        Args:
            bending_direction (str): Direction of the bending, must be one of:
                'x': when bending vector is in the positive x direction (bencing about the x-axis causing compression on the positive y side)
                '-x': when bending vector is in the negative x direction (bending about the x-axis causing compression on the negative y side)
                'y': when bending vector is in the positive y direction (bending about the y-axis causing compression on the positive x side)
                '-y': when bending vector is in the negative y direction (bending about the y-axis causing compression on the negative x side)

        Returns:
            float: Distance from extreme compression fiber to centroid of tension reinforcement.
            None: If no rebars are defined.

        Raises:
            ValueError: If direction is not one of the valid options.
        """
        
        if not self.rebars:
            return None
            
        valid_directions = ['x', '-x', 'y', '-y']
        if bending_direction not in valid_directions:
            raise ValueError(f"Direction must be one of {valid_directions}")
            
        if bending_direction == 'x':
            return abs(self.section_max_x - self.rebars_min_x)
        elif bending_direction == '-x':
            return abs(self.section_min_x - self.rebars_max_x)
        elif bending_direction == 'y':
            return abs(self.section_max_y - self.rebars_min_y)
        elif bending_direction == '-y':
            return abs(self.section_min_y - self.rebars_max_y)

    def compressive_effective_depth(self, bending_direction: str):
        """Computes the distance from extreme compression fiber to 
        centroid of longitudinal compression reinforcement.
        If no rebar is defined, it returns None.
        
        Args:
            bending_direction (str): Direction of the bending, must be one of:
                'x': when bending vector is in the positive x direction (bencing about the x-axis causing compression on the positive y side)
                '-x': when bending vector is in the negative x direction (bending about the x-axis causing compression on the negative y side)
                'y': when bending vector is in the positive y direction (bending about the y-axis causing compression on the positive x side)
                '-y': when bending vector is in the negative y direction (bending about the y-axis causing compression on the negative x side)

        Returns:
            float: Distance from extreme compression fiber to centroid of tension reinforcement.
            None: If no rebars are defined.

        Raises:
            ValueError: If direction is not one of the valid options.
        """
        
        if not self.rebars:
            return None
            
        valid_directions = ['x', '-x', 'y', '-y']
        if bending_direction not in valid_directions:
            raise ValueError(f"Direction must be one of {valid_directions}")
            
        if bending_direction == 'x':
            return abs(self.section_max_x - self.rebars_max_x)
        elif bending_direction == '-x':
            return abs(self.section_min_x - self.rebars_min_x)
        elif bending_direction == 'y':
            return abs(self.section_max_y - self.rebars_max_y)
        elif bending_direction == '-y':
            return abs(self.section_min_y - self.rebars_min_y)        
        
    def tBar_area(self, direction:str) -> float:
        """Returns the total area of transverse bars in the section."""
        if not self.tBars:
            return 0.0
        
        direction = check_direction_validity(direction)
        
        if direction in ['x', '-x']:
            return sum(tBar.Ax for tBar, _ in self.tBars)
        elif direction in ['y', '-y']:
            return sum(tBar.Ay for tBar, _ in self.tBars)
    
    def plot(self, ax: axs.Axes = None, **kwargs):
        """Plots the wall and its rebars.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes are created.
            **kwargs: Additional keyword arguments for the plot.
        """
        figsize = kwargs.get('figsize', (8, 6))
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        wall = self.shape
        wall.plot(
            ax=ax, 
            facecolor=kwargs.get('con_facecolor', 'lightgray'),
            edgecolor=kwargs.get('con_edgecolor', 'black'),
            linewidth=kwargs.get('con_linewidth', 1),
            alpha=kwargs.get('con_alpha', 0.5),
            centroid_color=kwargs.get('con_centroid_color', 'red'),
            labelsize=kwargs.get('con_ticksize', 8),
            zorder = kwargs.get('con_zorder', 1),
            figsize=figsize
        )
        
        rebar_color = kwargs.get('rebar_color', 'black')
        for rebar in self.rebars:
            rebar.plot(
                ax=ax,             
                facecolor=kwargs.get('bar_facecolor'), 
                edgecolor=kwargs.get('bar_edgecolor', 'black'),
                linewidth=kwargs.get('bar_linewidth', 1),
                alpha=kwargs.get('bar_alpha', 1.0),
                zorder= kwargs.get('bar_zorder', 10),
            )
            
        ax.relim()
        ax.autoscale_view()
        ax.set_aspect('equal')
        return ax
        
        
# Functions =====================================================
def get_reference_point(concrete_sec: RectangleConcreteSection, bending_direction: str) -> Point:
    """Returns a reference point for the bending direction in a rectangular concrete section.
    The reference point is the location of the fiber of maximum compressive strain.

    Args:
        concrete_sec (RectangleConcreteSection): Concrete section object containing the rebars.
        bending_direction (str): Direction of the bending, must be one of:
            'x': when bending vector is in the positive x direction (bencing about the x-axis causing compression on the positive y side)
            '-x': when bending vector is in the negative x direction (bending about the x-axis causing compression on the negative y side)
            'y': when bending vector is in the positive y direction (bending about the y-axis causing compression on the positive x side)
            '-y': when bending vector is in the negative y direction (bending about the y-axis causing compression on the negative x side)

    Returns:
        Point: Reference point for the bending direction.
    """
    REF_POINTS = {
        'x' : Point(concrete_sec.center.x, concrete_sec.section_min_y),
        '-x': Point(concrete_sec.center.x, concrete_sec.section_max_y),
        'y' : Point(concrete_sec.section_max_x, concrete_sec.center.y),
        '-y': Point(concrete_sec.section_min_x, concrete_sec.center.y)
    }
    
    return REF_POINTS.get(bending_direction, None)
    
    
    
def get_rebar_forces(
    concrete_sec: RectangleConcreteSection, 
    neutral_depth:Union[float, np.ndarray], 
    bending_direction: str
) -> Tuple[List[float], List[float], List[float], List[float]]:
    
    """Computes the strain, stress, and force for each rebar in the concrete_sec.

    Args:
        concrete_sec (RectangleConcreteSection): Concrete section object containing the rebars.
        neutral_depth Union[float, np.ndarray]: Depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.
        bending_direction (str): Direction of the bending, must be one of:
            'x': when bending vector is in the positive x direction (bencing about the x-axis causing compression on the positive y side)
            '-x': when bending vector is in the negative x direction (bending about the x-axis causing compression on the negative y side)
            'y': when bending vector is in the positive y direction (bending about the y-axis causing compression on the positive x side)
            '-y': when bending vector is in the negative y direction (bending about the y-axis causing compression on the negative x side)

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: Tuple of lists containing rebar depth, strain, stress, and force.
    """
    if not concrete_sec.rebars:
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    bending_direction = check_direction_validity(bending_direction)
            
    if not isinstance(neutral_depth, np.ndarray):
        neutral_depth = np.array([neutral_depth])
        
    # Calculate curvature based on neutral depth
    try:
        curvature = concrete_sec.mat.eps_u / neutral_depth
    except ZeroDivisionError:
        raise ValueError("Neutral depth cannot be zero.")
    
    # Compute the total height of the section and the depth of each rebar based on bending direction
    if bending_direction in ['x', '-x']:
        section_height = concrete_sec.section_max_y - concrete_sec.section_min_y
        rebar_depth = np.array([rebar.center.y for rebar in concrete_sec.rebars]).reshape(-1, 1)
        depth_multiplier = 1 if bending_direction == '-x' else -1
    else: # bending_direction in ['y', '-y']
        section_height = concrete_sec.section_max_x - concrete_sec.section_min_x
        rebar_depth = np.array([rebar.center.x for rebar in concrete_sec.rebars]).reshape(-1, 1)
        depth_multiplier = 1 if bending_direction == '-y' else -1
    
    # Calculate strain, stress, and force for each rebar
    strain = curvature.T * (rebar_depth * depth_multiplier - (section_height / 2 - neutral_depth.T))
    stress = np.array([rebar.mat.stress(eps) for rebar, eps in zip(concrete_sec.rebars, strain)])
    force = stress * np.array([rebar.area for rebar in concrete_sec.rebars]).reshape(-1, 1)
    
    return rebar_depth, strain, stress, force
    
    
def get_rect_concrete_force(
    concrete_sec: 'RectangleConcreteSection', 
    neutral_depth: Union[float, np.ndarray], 
    bending_direction: str,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """Calculate forces in a rectangular concrete section due to bending.
    
    Args:
        concrete_sec: Rectangular concrete section object
        neutral_depth Union[float, np.ndarray]: Depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.
        bending_direction (str): Direction of the bending, must be one of:
            'x': when bending vector is in the positive x direction (bencing about the x-axis causing compression on the positive y side)
            '-x': when bending vector is in the negative x direction (bending about the x-axis causing compression on the negative y side)
            'y': when bending vector is in the positive y direction (bending about the y-axis causing compression on the positive x side)
            '-y': when bending vector is in the negative y direction (bending about the y-axis causing compression on the negative x side)
        n_points: Number of points to discretize the section
        
    Returns:
        Tuple containing:
        - y_coordinates: Coordinates of the discretized strips
        - strain: Strain at each strip
        - stress: Stress at each strip
        - force: Force at each strip
    """
        
    bending_direction = check_direction_validity(bending_direction)
        
    if not isinstance(neutral_depth, np.ndarray):
        neutral_depth = np.array([neutral_depth])
        
    # Calculate curvature based on neutral depth
    try:
        curvature = concrete_sec.mat.eps_u / neutral_depth
    except ZeroDivisionError:
        raise ValueError("Neutral depth cannot be zero.")        
    
    # Calculate section dimensions based on bending direction
    if bending_direction in ['x', '-x']:
        section_height = concrete_sec.section_max_y - concrete_sec.section_min_y
        section_width = concrete_sec.section_max_x - concrete_sec.section_min_x
        depth_multiplier = 1 if bending_direction == '-x' else -1
    else:  # bending_direction in ['y', '-y']
        section_height = concrete_sec.section_max_x - concrete_sec.section_min_x
        section_width = concrete_sec.section_max_y - concrete_sec.section_min_y
        depth_multiplier = 1 if bending_direction == '-y' else -1
          
    # Calculate coordinates for discretization
    depth_coordinates = np.linspace(-section_height/2, section_height/2, n_points+1).reshape(-1, 1)
    strip_thickness = section_height / n_points
    
    # Compute strain and stress at the center of each strip
    strip_center_depth = depth_coordinates[:-1] + strip_thickness / 2
    strain = curvature.T * (strip_center_depth * depth_multiplier - (section_height/2 - neutral_depth.T))
    stress = concrete_sec.mat.stress(strain)

    # Create concrete strips based on section orientation
    if bending_direction in ['x', '-x']:
        concrete_strips = [
            sh.Rectangle(section_width, strip_thickness, (concrete_sec.center.x, depth[0])) 
            for depth in strip_center_depth
        ]
    else: # bending_direction in ['y', '-y']:
        concrete_strips = [
            sh.Rectangle(strip_thickness, section_width, (depth[0], concrete_sec.center.y)) 
            for depth in strip_center_depth
        ]

    # Calculate net areas of strips accounting for rebar holes
    rebar_shapes = [rebar.shape for rebar in concrete_sec.rebars]
    net_areas = sh.get_rectangle_net_area(concrete_strips, rebar_shapes, show_plot=False)
    
    # Calculate force in each strip
    force = stress * net_areas.reshape(-1, 1)
    
    return strip_center_depth, strain, stress, force


def get_section_internal_force(concrete_sec: RectangleConcreteSection, curvature:Union[float, np.ndarray], y_neutral:Union[float, np.ndarray]) -> float:
    _, _, _, rebar_forces = get_rebar_forces(concrete_sec, curvature, y_neutral)
    _ , _, _, concrete_forces = get_rect_concrete_force(concrete_sec, curvature, y_neutral)
    total_rebar_force = rebar_forces.sum()
    total_concrete_force = concrete_forces.sum()
    
    return total_rebar_force + total_concrete_force  


def get_section_internal_moment(concrete_sec: RectangleConcreteSection, curvature:Union[float, np.ndarray], y_neutral:Union[float, np.ndarray]) -> float:
    y_bar, _, _, f_bar = get_rebar_forces(concrete_sec, curvature, y_neutral)
    y_con, _, _, f_con = get_rect_concrete_force(concrete_sec, curvature, y_neutral)
    total_bar_moment = (y_bar * f_bar).sum(axis=0)
    total_con_moment = (y_con * f_con).sum(axis=0)
    
    return total_bar_moment + total_con_moment


def compute_neutral_axis(concrete_sec: RectangleConcreteSection, external_force:float, method:str, **kwargs) -> float: 
    xtol = kwargs.get('xtol', 1e-2)
    maxiter = kwargs.get('maxiter', 100)
    h = max(concrete_sec.shape.height, concrete_sec.shape.width)
    x0 = kwargs.get('x0', 1e-2)
    x1 = kwargs.get('x1', h)
    
    def equilibrium(y_neutral: Union[float, np.ndarray]) -> float:    
        curvature = concrete_sec.mat.eps_u / y_neutral
        internal_force = get_section_internal_force(concrete_sec, curvature, y_neutral)
        return internal_force - external_force

    if method in ['bisect', 'brentq']:
        result = root_scalar(equilibrium, bracket=[x0, x1], method=method, xtol=xtol, maxiter=maxiter)
    elif method in ['secant', 'newton']:
        result = root_scalar(equilibrium, method=method, x0=x0, x1=x1, xtol=xtol, maxiter=maxiter)
        
    if result.converged:
        return result.root
    else:
        raise ValueError("Failed to find a solution for the neutral axis depth.")


def outer_most_rebars(con: RectangleConcreteSection) -> List[RebarSection]:
    """This function returns the outer most rebars in the section.
    The outer most rebars are those with the maximum y center coordinate (or x, depending on the orientation).
    There could be more than one rebar with the same y center coordinate.
    Only works when the section orientation is either vertical or horizontal.
    If the section is a square, it will return the rebars with the minimum y center coordinate.

    Args:
        con (RectangleConcreteSection): Concrete section object.

    Returns:
        List[RebarSection]: List of outer most rebars.
    """
    rebars = con.rebars
    
    if con.shape.height >= con.shape.width:
        y_min = min([rebar.center.y for rebar in rebars])
        return [rebar for rebar in rebars if rebar.center.y == y_min]
    else:
        x_min = min([rebar.center.x for rebar in rebars])
        return [rebar for rebar in rebars if rebar.center.x == x_min]


def min_depth_of_neutral_axis(con: RectangleConcreteSection) -> float:
    """This function computes the minimum depth of the neutral axis.
    The minimum depth of the neutral axis is measured from the outer most compressive concrete fiber.
    The minimum neutral axis depth occurs when the outer most tensile rebar reaches its ultimate strain (eps_su).

    Args:
        con (RectangleConcreteSection): Concrete section object.

    Returns:
        float: minimum depth of the neutral axis.
    """
    # Get the outer most tensile rebars
    rebars = outer_most_rebars(con)
    
    # Minimum ultimate strain among the outer most tensile rebars (positive)
    eps_su = min([rebar.mat.eps_u for rebar in rebars])
    
    # Ultimate strain of the concrete (negative)
    eps_cu = con.mat.eps_u
    
    y_total = max(con.shape.height, con.shape.width)
    
    return -eps_cu / (-eps_cu + eps_su) * y_total
    

def nominal_tensile_strength(concrete_sec: RectangleConcreteSection) -> float:
    """Calculates the nominal tensile strength of the concrete section per ACI 318-25: 22.4.3.
    The nominal tensile strength is the sum of the yield strengths of all the rebars in the section.

    Args:
        concrete_sec (RectangleConcreteSection): Concrete section object.

    Returns:
        float: Nominal tensile strength of the section in lbs.
    """
    strength = 0.0
    for rebar in concrete_sec.rebars:
        strength += rebar.mat.fy * rebar.area
        
    return strength
        
        
def nominal_compressive_strength(concrete_sec: RectangleConcreteSection) -> float:
    """Calculates the nominal compressive strength of the concrete section per ACI 318-25: 22.4.2..
    The nominal compressive strength is the sum of the compressive strength of the concrete and the yield strengths of all the rebars in the section.

    Args:
        concrete_sec (RectangleConcreteSection): Concrete section object.

    Returns:
        float:  Nominal compressive strength of the section in lbs.
    """
    strength = 0.85 * concrete_sec.mat.fc * concrete_sec.net_area
    
    for rebar in concrete_sec.rebars:
        strength += rebar.mat.fy * rebar.area
    
    return -1 * strength
    

def max_nominal_compressive_strength(concrete_sec: RectangleConcreteSection) -> float:
    """Calculates the maximum nominal compressive strength of the concrete section per ACI 318-25: 22.4.2.1.
    The reduction factor is to account for accidnetal eccentricity in the section.

    Args:
        concrete_sec (RectangleConcreteSection): Concrete section object.

    Returns:
        float: Maximum nominal compressive strength of the section in lbs.
    """
    return 0.80 * nominal_compressive_strength(concrete_sec)


def generate_force_moment_interaction(con: RectangleConcreteSection, n_points:int=100, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_total = max(con.shape.height, con.shape.width)
    
    y_neutral_min = min_depth_of_neutral_axis(con)
    y_neutral_max = kwargs.get('ynmax_multiplier', 1.0) * y_total
    y_neutral = np.linspace(y_neutral_min, y_neutral_max, n_points).reshape(-1, 1)
    
    curvature = con.mat.eps_u / y_neutral
    
    _, _, _, f_bar = get_rebar_forces(con, curvature, y_neutral)
    _, _, _, f_con = get_rect_concrete_force(con, curvature, y_neutral)
    
    # Calculate the total force and moment in the section
    f_total = f_bar.sum(axis=0) + f_con.sum(axis=0)
    m_total = get_section_internal_moment(con, curvature, y_neutral)
    
    # Calculate section forces and moments under pure tension and compression
    tensile_strength = nominal_tensile_strength(con)
    compressive_strength = nominal_compressive_strength(con)
    
    # Prepend (np.inf, tensile_strength, 0)
    y_neutral = np.vstack(([np.inf], y_neutral))
    f_total = np.hstack(([tensile_strength], f_total))
    m_total = np.hstack(([0], m_total))

    # Append (np.inf, compressive_strength, 0)
    y_neutral = np.vstack((y_neutral, [np.inf]))
    f_total = np.hstack((f_total, [compressive_strength]))
    m_total = np.hstack((m_total, [0]))
    
    return y_neutral, f_total, m_total
    

def get_tensile_and_compressive_rebars(concrete_sec: RectangleConcreteSection, neutral_depth: float, bending_direction:str) -> Tuple[List[RebarSection]]:
    
    if not concrete_sec.rebars:
        return ([], [])
    
    bending_direction = check_direction_validity(bending_direction)
        
    if bending_direction == '-x':
        tensile_rebars = [rebar for rebar in concrete_sec.rebars if rebar.center.y < concrete_sec.section_max_y - neutral_depth]
    elif bending_direction == 'x':
        tensile_rebars = [rebar for rebar in concrete_sec.rebars if rebar.center.y > concrete_sec.section_min_y + neutral_depth]
    elif bending_direction == 'y':
        tensile_rebars = [rebar for rebar in concrete_sec.rebars if rebar.center.x < concrete_sec.section_max_x - neutral_depth]
    elif bending_direction == '-y':
        tensile_rebars = [rebar for rebar in concrete_sec.rebars if rebar.center.x > concrete_sec.section_min_x + neutral_depth]
    
    compresive_rebars = [rebar for rebar in concrete_sec.rebars if rebar not in tensile_rebars]
    
    return tensile_rebars, compresive_rebars
        

def get_rebars_centroids(concrete_sec: RectangleConcreteSection, neutral_depth: float, bending_direction: str) -> Optional[Tuple[Point, Point]]:
    """Computes the centroids of tensile and compressive rebars in a concrete section based on the neutral depth and bending direction.

    Args:
        concrete_sec (RectangleConcreteSection): Concrete section object containing the rebars.
        neutral_depth (float): Depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.
        bending_direction (str): Direction of the bending, must be one of:
            'x': when bending vector is in the positive x direction (bending about the x-axis causing compression on the positive y side)
            '-x': when bending vector is in the negative x direction (bending about the x-axis causing compression on the negative y side)
            'y': when bending vector is in the positive y direction (bending about the y-axis causing compression on the positive x side)
            '-y': when bending vector is in the negative y direction (bending about the y-axis causing compression on the negative x side)

    Returns:
        Optional[Tuple[Point, Point]]: A tuple containing the centroids of tensile and compressive rebars as Point objects.
        Returns None if no rebars are defined or if the centroids cannot be computed.
    """
    
    bending_direction = check_direction_validity(bending_direction)
    
    # Compute the strain, stress, and force for all rebars based on the neutral depth and bending direction
    coord, _, _, force = get_rebar_forces(concrete_sec, neutral_depth, bending_direction)
    
    # Compute total tensile rebar force
    mask_st = force > 0
    f_st = force[mask_st].sum()

    # Compute centroid of tensile rebar forces
    tensile_centroid = np.sum(coord[mask_st] * force[mask_st]) / f_st if f_st != 0 else None
    
    # Compute total compressive rebar force
    mask_sc = force < 0
    f_sc = force[mask_sc].sum()

    # Compute centroid of compressive rebar forces
    compressive_centroid = np.sum(coord[mask_sc] * force[mask_sc]) / f_sc if f_sc != 0 else None
    
    if bending_direction in ['x', '-x']:
        if tensile_centroid is not None and compressive_centroid is not None:
            return (
                Point(concrete_sec.center.x, tensile_centroid),
                Point(concrete_sec.center, compressive_centroid)
            )
        else:
            return None
    elif bending_direction in ['y', '-y']:
        if tensile_centroid is not None and compressive_centroid is not None:
            return (
                Point(tensile_centroid, concrete_sec.center.y),
                Point(compressive_centroid, concrete_sec.center.y)
            )


def get_tensile_effective_depth(concrete_sec: RectangleConcreteSection, neutral_depth:float, bending_direction: str) -> float:
    """Computes the distance from extreme compression fiber to centroid of longitudinal tension reinforcement.
        If no rebar is defined, it returns None.
        
        Args:
            concrete_sec (RectangleConcreteSection): Concrete section object.
            neutral_depth (float): Depth of the neutral axis from the fiber of maximum compressive strain considering the bending direction.
            bending_direction (str): Direction of the bending, must be one of:
                'x': when bending vector is in the positive x direction (bencing about the x-axis causing compression on the positive y side)
                '-x': when bending vector is in the negative x direction (bending about the x-axis causing compression on the negative y side)
                'y': when bending vector is in the positive y direction (bending about the y-axis causing compression on the positive x side)
                '-y': when bending vector is in the negative y direction (bending about the y-axis causing compression on the negative x side)

    Returns:
        float: Distance from extreme compression fiber to centroid of tension reinforcement.
        None: If no rebars are defined or if the centroid cannot be computed.
    """
    bending_direction = check_direction_validity(bending_direction)
    if not concrete_sec.rebars:
        return None

    tensile_rebar_centroid, _ = get_rebars_centroids(concrete_sec, neutral_depth, bending_direction)
    
    if tensile_rebar_centroid is None:
        return None
    
    # Get the point of maximum compressive strain based on the bending direction
    ref_point = get_reference_point(concrete_sec, bending_direction)
    
    return tensile_rebar_centroid.distance_to(ref_point[bending_direction])

    

# Functions =============================================================
def check_direction_validity(direction: str) -> str:
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


# Main function =========================================================      
def main():
    pass
    
    
if __name__ == "__main__":
    main()