
from copy import deepcopy
import re
from shapes import *
from materials import ACIConcrete, Concrete
from steel import RebarSection
from typing import Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt



class RectangleWall:
    def __init__(self, shape: Rectangle, concrete_mat: Concrete, cover: float, center: Union[Point, Tuple[float, float]] = (0.0, 0.0)):
        self._shape = deepcopy(shape)
        self._shape.center = center
        self.mat = concrete_mat
        if shape.width - 2 * cover <= 0 or shape.height - 2 * cover <= 0:
            raise ValueError("Cover is too large for the wall dimensions.")
        self.cover = cover
        self.rebars: List[RebarSection] = []
        
    @property
    def shape(self) -> Rectangle:
        return self._shape
    
    @property
    def center(self) -> Tuple:
        return self._shape.center[0], self._shape.center[1]
    
    @center.setter
    def center(self, new_center: Union[Point, Tuple[float, float]]):
        if isinstance(new_center, tuple):
            new_center = Point(new_center[0], new_center[1])
        self._shape.center = new_center
        
    @property
    def wall_minus_corner(self) -> Rectangle:
        return Rectangle(self._shape.width - 2 * self.cover, self._shape.height - 2 * self.cover, self._shape.center)

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
        
        r = deepcopy(rebar)
        r.center = center
        
        # Check if the new rebar overlaps with existing rebars
        for existing_rebar in self.rebars:
            if circles_overlap(r.shape, existing_rebar.shape):
                raise ValueError(f"Rebar at {r.center} overlaps with existing rebar at {existing_rebar.center}.")
        
        # Check if the rebar is within the wall bounds minus cover
        if not circle_within_rectangle(r.shape, self.wall_minus_corner):
            raise ValueError(f"Rebar at {r.center} is overlapping the cover region or crossing the wall bounds.")
        
        self.rebars.append(r)
        
    def add_rebar_row(self, rebar:RebarSection, center: Union[Point, Tuple[float, float]], n: int, spacing: float, reverse: bool = False):
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
                center = Point(center[0], center[1])
        
        for i in range(n):
            offset = i * spacing
            if reverse:
                offset = -offset
            
            if self.shape.width > self.shape.height:
                new_center = Point(center.x + offset, center.y)
            else:
                new_center = Point(center.x, center.y + offset)
            
            self.add_rebar(rebar, new_center)

    def plot(self, ax: matplotlib.axes.Axes = None, **kwargs):
        """Plots the wall and its rebars.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes are created.
            **kwargs: Additional keyword arguments for the plot.
        """
        fig_size = kwargs.get('figsize', (8, 6))
        
        if ax is None:
            fig, ax = plt.subplots(figsize=fig_size)
        
        wall = self.shape
        wall_fill_color = kwargs.get('wall_fill_color', 'lightgray')
        wall.plot(ax=ax, fill_color=wall_fill_color, figsize=fig_size, **kwargs)
        
        rebar_color = kwargs.get('rebar_color', 'black')
        for rebar in self.rebars:
            rebar.shape.plot(ax=ax, color=rebar_color, **kwargs)
        
        ax.relim()
        ax.autoscale_view()
        ax.set_aspect('equal')
        return ax
        
        
# Utility functions =====================================================

def get_rebar_forces(concrete_sec: RectangleWall, curvature:float, y_neutral:float) -> Tuple[List[float], List[float], List[float]]:
    """Computes the strain, stress, and force for each rebar in the concrete_sec.

    Args:
        concrete_sec (RectangleWall): Concrete section object containing the rebars.
        curvature (float): Curvature of the section.
        y_neutral (float): Depth of the neutral axis from the fiber of maximum compressive strain.

    Returns:
        Tuple[List[float], List[float], List[float]]: Tuple of lists containing strain, stress, and force for each rebar.
    """
    y_total = max(concrete_sec.shape.height, concrete_sec.shape.width)
    y_bar, strain, stress, force = [], [], [], []
    
    for i, rebar in enumerate(concrete_sec.rebars):
        y_bar.append(rebar.center.y if concrete_sec.shape.width < concrete_sec.shape.height else rebar.center.x)
        strain.append(-1 * curvature * (y_bar[i] - (y_total / 2 - y_neutral)))
        stress.append(rebar.mat.stress(strain[i]))
        force.append(stress[i] * rebar.area)
        
    return y_bar, strain, stress, force
    
def get_rect_concrete_force(concrete_sec: RectangleWall, curvature:float, y_neutral:float) -> float:
    y_total = max(concrete_sec.shape.height, concrete_sec.shape.width)
    sec_width = min(concrete_sec.shape.width, concrete_sec.shape.height)
    y_con = np.linspace(-y_total / 2, y_total / 2, 100).reshape(-1, 1)
    delta_y = y_total / y_con.shape[0]
    strain =  -1 * curvature * (y_con - (y_total / 2 - y_neutral))
    stress = concrete_sec.mat.stress(strain)
    force = stress * sec_width * delta_y
    
    return y_con, strain, stress, force



  
# Main function =========================================================      
def main():
    pass
    
    
if __name__ == "__main__":
    main()