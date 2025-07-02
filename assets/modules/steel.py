##############################################
# Steel Section Module
# Written by: Hossein Karagah
# Date: 2024-10-05
# Description: This module defines the steel section classes representing geometric properties used in engineering calculation.
##############################################


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes
from assets.modules.shapes import Circle, Point
from assets.modules.materials import BilinearSteel
from typing import Union, Tuple
import numpy as np
from math import pi

#############################################
# Global variables
#############################################

_REBAR_DIAMETERS = {
    3: 0.375,
    4: 0.500,
    5: 0.625,
    6: 0.750,
    7: 0.875,
    8: 1.000,
    9: 1.128,
    10: 1.270,
    11: 1.410,
    14: 1.693,
    18: 2.257,
}



# Main Classes #############################################

class LongitudinalRebar:
    """ Represents a longitudinal rebar section.
    This class is used to define a longitudinal rebar section, which can be used in concrete section analysis and design."""
    
    def __init__(self, size: int, mat: BilinearSteel, center: Union[Point, tuple[float, float]] = (0.0, 0.0), is_skin_bar: bool = False):
        """ Initializes a longitudinal rebar section.
        This class represents a longitudinal rebar section, which can be used in concrete section analysis and design.

        Args:
            size (int): The size of the rebar (e.g., #4, #5, etc.).
            mat (BilinearSteel): The material properties of the rebar.
            center (Union[Point, tuple[float, float]], optional): The center point of the rebar section. Defaults to (0.0, 0.0).
            is_skin_bar (bool, optional): If True, the rebar is considered a skin bar and does not contribute to effective depth calculation. Defaults to False.
        """
        self._size = size
        self._mat = mat
        if isinstance(center, tuple):
            center = Point(*center)
        self._shape = Circle(radius = self.dia / 2, center = center)
        self._is_skin_bar = is_skin_bar
        
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def mat(self) -> BilinearSteel:
        return self._mat
    
    @property
    def shape(self) -> Circle:
        return self._shape
    
    @property
    def dia(self) -> float:
        return get_rebar_diameter(self._size)
    
    @property
    def area(self) -> float:
        return self._shape.area
    
    @property
    def center(self) -> Point:
        return self._shape.center
    
    @center.setter
    def center(self, new_center: Union[Point, tuple[float, float]]):
        if isinstance(new_center, tuple):
            new_center = Point(*new_center)
        self._shape.center = new_center
        
    @property
    def is_skin_bar(self) -> bool:
        return self._is_skin_bar
    
    def __repr__(self):
        return (f"#{self._size} long. rebar at {self._shape.center}, fy={self._mat.fy}{', skin bar' if self._is_skin_bar else ''}")
    
    def __str__(self):
        return (f"#{self._size} long. rebar at {self._shape.center}, fy={self._mat.fy}{', skin bar' if self._is_skin_bar else ''}")

    def plot(self, ax:matplotlib.axes.Axes=None, **kwargs):
        """
        Plots the rebar section on the given axis.
        
        Args:
            ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure and axis will be created.
            **kwargs: Additional keyword arguments for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        self.shape.plot(
            ax=ax,
            facecolor=kwargs.get('facecolor'), 
            edgecolor=kwargs.get('edgecolor', 'black'),
            linewidth=kwargs.get('linewidth', 1),
            alpha=kwargs.get('alpha'),
            centroid_color=kwargs.get('centroid_color'),
            zorder=kwargs.get('zorder'),
        )
        
        return ax

class TransverseRebar:
    def __init__(self, size: int, mat: BilinearSteel, nLegsX: int, nLegsY: int, spacing: float, is_looped: bool = True, is_spiral: bool = False):
        """ Initializes a transverse rebar section.
        This class represents a transverse rebar section (tie, stirrups, or spiral), which can be used in concrete section analysis and design.

        Args:
            size (int): The size of the rebar (e.g., #4, #5, etc.).
            mat (BilinearSteel): The material properties of the rebar.
            nLegsX (int): number of legs in the X direction.
            nLegsY (int): number of legs in the Y direction.
            spacing (float): spacing between the transverse rebars.
            is_looped (bool, optional): Whether the transverse rebar is looped or single-legged. Defaults to True.
            is_spiral (bool, optional): Whether the transverse rebar is a spiral or not. Defaults to False.
        """
        self._size = size
        self._mat = mat
        self._nLegsX = nLegsX
        self._nLegsY = nLegsY
        self._spacing = spacing
        self._is_looped = is_looped
        if is_spiral and not is_looped:
            raise ValueError("Spiral transverse rebars must be looped.")
        self._is_spiral = is_spiral
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def mat(self) -> BilinearSteel:
        return self._mat
    
    @property
    def dia(self) -> float:
        return get_rebar_diameter(self._size)
    
    @property
    def spacing(self) -> float:
        return self._spacing
    
    @property
    def nLegsX(self) -> int:
        return self._nLegsX
    
    @property
    def nLegsY(self) -> int:
        return self._nLegsY
        
    @property
    def Ax(self) -> float:
        return (pi * self.dia ** 2 / 4) * self.nLegsX
    
    @property
    def Ay(self) -> float:
        return (pi * self.dia ** 2 / 4) * self.nLegsY
    
    @property
    def is_looped(self) -> bool:
        return self._is_looped
    
    @property
    def is_spiral(self) -> bool:
        return self._is_spiral
    
    def __repr__(self):
        return (f"#{self._size} {'looped' if self._is_looped else 'single-legged'} trans. rebar {'(spiral)' if self._is_spiral else ''}, fy={self.mat.fy}, nLegsX={self._nLegsX}, nLegsY={self._nLegsY}, spacing={self._spacing}")
        
    def __str__(self):
        return (f"#{self._size} {'looped' if self._is_looped else 'single-legged'} trans. rebar {'(spiral)' if self._is_spiral else ''}, fy={self.mat.fy}, nLegsX={self._nLegsX}, nLegsY={self._nLegsY}, spacing={self._spacing}")
    

        
        

# Functions #############################################

def get_rebar_diameter(size: int) -> float:
    """Returns the diameter of the rebar in inches based on its size."""
    try:
        return _REBAR_DIAMETERS[size]
    except KeyError:
        raise ValueError(f"Invalid rebar size: {size}. Valid sizes are {sorted(_REBAR_DIAMETERS.keys())}.")



# Main Function #########################################
def main():
    pass
    
    
if __name__ == "__main__":
    main()