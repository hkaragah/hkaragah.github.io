################################
# Concrete Section Module
# Written by: Hossein Karagah
# Date: 2025-03-10
# Description: This module provides classes and functions to model and analyze rectangular concrete sections with rebars.
################################


from copy import deepcopy
import shapes as sh
from materials import ACIConcrete, Concrete
from steel import RebarSection
from typing import Tuple, List, Union
import numpy as np
import matplotlib.axes as axs
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, fsolve


class RectangleConcreteSection:
    def __init__(self, shape: sh.Rectangle, concrete_mat: Concrete, cover: float, center: Union[sh.Point, Tuple[float, float]] = (0.0, 0.0)):
        self._shape = deepcopy(shape)
        self._shape.center = center
        self.mat = concrete_mat
        if shape.width - 2 * cover <= 0 or shape.height - 2 * cover <= 0:
            raise ValueError("Cover is too large for the wall dimensions.")
        self.cover = cover
        self.rebars: List[RebarSection] = []
        
    @property
    def shape(self) -> sh.Rectangle:
        return self._shape
    
    @property
    def gross_area(self) -> float:
        return self._shape.area
    
    @property
    def net_area(self) -> float:
        rebar_area = sum([rebar.area for rebar in self.rebars])
        return self.gross_area - rebar_area
    
    @property
    def center(self) -> Tuple:
        return (self._shape.center.x, self._shape.center.y)
    
    @center.setter
    def center(self, new_center: Union[sh.Point, Tuple[float, float]]):
        if isinstance(new_center, tuple):
            new_center = sh.Point(new_center[0], new_center[1])
        self._shape.center = new_center
        
    @property
    def wall_minus_corner(self) -> sh.Rectangle:
        return sh.Rectangle(self._shape.width - 2 * self.cover, self._shape.height - 2 * self.cover, self._shape.center)

    def add_rebar(self, rebar: RebarSection, center: Union[sh.Point, Tuple[float, float]]):
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
            center = sh.Point(center[0], center[1])
        
        r = deepcopy(rebar)
        r.center = center
        
        # Check if the new rebar overlaps with existing rebars
        for existing_rebar in self.rebars:
            if sh.circles_overlap(r.shape, existing_rebar.shape):
                raise ValueError(f"Rebar at {r.center} overlaps with existing rebar at {existing_rebar.center}.")
        
        # Check if the rebar is within the wall bounds minus cover
        if not sh.circle_within_rectangle(r.shape, self.wall_minus_corner):
            raise ValueError(f"Rebar at {r.center} is overlapping the cover region or crossing the wall bounds.")
        
        self.rebars.append(r)
        
    def add_rebar_row(self, rebar:RebarSection, center: Union[sh.Point, Tuple[float, float]], n: int, spacing: float, reverse: bool = False):
        """Adds a row of rebars to the wall.
        The row is added at the specified center point. The function checks if the new rebars overlap with existing rebars and if they are within the wall bounds minus cover.

        Args:
            rebar (RebarSection): RebarSection object to be added.
            center (Union[Point, Tuple[float, float]]): Center point of the row.
                If a tuple is provided, it will be converted to a Point object.
            n (int): Number of rebars in the row.
            spacing (float): Spacing between the rebars.
                The spacing is measured from the center of one rebar to the center of the next rebar.
            reverse (bool): If True, the row is added in backward order from the provided center.
        """
        if isinstance(center, tuple):
                center = sh.Point(center[0], center[1])
        
        for i in range(n):
            offset = i * spacing
            if reverse:
                offset = -offset
            
            if self.shape.width > self.shape.height:
                new_center = sh.Point(center.x + offset, center.y)
            else:
                new_center = sh.Point(center.x, center.y + offset)
            
            self.add_rebar(rebar, new_center)

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
        
        
# Utility functions =====================================================

def get_rebar_forces(concrete_sec: RectangleConcreteSection, curvature:Union[float, np.ndarray], y_neutral:Union[float, np.ndarray]) -> Tuple[List[float], List[float], List[float]]:
    """Computes the strain, stress, and force for each rebar in the concrete_sec.

    Args:
        concrete_sec (RectangleConcreteSection): Concrete section object containing the rebars.
        curvature (float): Curvature of the section.
        y_neutral (float): Depth of the neutral axis from the fiber of maximum compressive strain.

    Returns:
        Tuple[List[float], List[float], List[float]]: Tuple of lists containing strain, stress, and force for each rebar.
    """
    if not isinstance(curvature, np.ndarray):
        curvature = np.array([curvature])
        
    if not isinstance(y_neutral, np.ndarray):
        y_neutral = np.array([y_neutral])
        
    is_taller = concrete_sec.shape.height > concrete_sec.shape.width
    y_total = max(concrete_sec.shape.height, concrete_sec.shape.width)
    y_bar = np.array([rebar.center.y if is_taller else rebar.center.x for rebar in concrete_sec.rebars]).reshape(-1, 1)
    strain = curvature.T * (y_bar - (y_total / 2 - y_neutral.T))
    stress = np.array([rebar.mat.stress(eps) for rebar, eps in zip(concrete_sec.rebars, strain)])
    force = stress * np.array([rebar.area for rebar in concrete_sec.rebars]).reshape(-1, 1)
    
    return y_bar, strain, stress, force
    
    
def get_rect_concrete_force(
    concrete_sec: 'RectangleConcreteSection', 
    curvature: Union[float, np.ndarray], 
    y_neutral: Union[float, np.ndarray], 
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate forces in a rectangular concrete section due to bending.
    
    Args:
        concrete_sec: Rectangular concrete section object
        curvature: Curvature value(s) of the section
        y_neutral: Neutral axis position(s)
        n_points: Number of points to discretize the section
        
    Returns:
        Tuple containing:
        - y_coordinates: Coordinates of the discretized strips
        - strain: Strain at each strip
        - stress: Stress at each strip
        - force: Force at each strip
    """
    # Convert inputs to numpy arrays if they aren't already
    curvature_array = np.atleast_1d(curvature)
    y_neutral_array = np.atleast_1d(y_neutral)
    
    # Determine section orientation and dimensions
    is_taller = concrete_sec.shape.height > concrete_sec.shape.width
    section_height = concrete_sec.shape.height
    section_width = concrete_sec.shape.width
    
    # Get the major and minor dimensions
    major_dimension = max(section_height, section_width)
    minor_dimension = min(section_height, section_width)
    
    # Calculate coordinates for discretization
    y_coordinates = np.linspace(-major_dimension/2, major_dimension/2, n_points).reshape(-1, 1)
    strip_thickness = major_dimension / n_points

    # Compute strain and stress for each strip
    strain = curvature_array.T * (y_coordinates - (major_dimension/2 - y_neutral_array.T))
    stress = concrete_sec.mat.stress(strain)

    # Create concrete strips based on section orientation
    if is_taller:
        # Vertical orientation (height > width)
        center_x = concrete_sec.center[0]
        concrete_strips = [
            sh.Rectangle(minor_dimension, strip_thickness, (center_x, y[0])) 
            for y in y_coordinates
        ]
    else:
        # Horizontal orientation (width > height)
        center_y = concrete_sec.center[1]
        concrete_strips = [
            sh.Rectangle(strip_thickness, minor_dimension, (x[0], center_y)) 
            for x in y_coordinates
        ]

    # Calculate net areas of strips accounting for rebar holes
    rebar_shapes = [rebar.shape for rebar in concrete_sec.rebars]
    net_areas = sh.get_rectangle_net_area(concrete_strips, rebar_shapes, show_plot=False)
    
    # Calculate force in each strip
    force = stress * net_areas.reshape(-1, 1)
    
    return y_coordinates, strain, stress, force


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
    
  
# Main function =========================================================      
def main():
    pass
    
    
if __name__ == "__main__":
    main()