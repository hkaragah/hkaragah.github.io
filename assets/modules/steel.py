import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes
from shapes import Circle, Point
from materials import BilinearSteel
from typing import Union, Tuple
import numpy as np


class RebarSection:
    def __init__(self, size: int, bar_mat: BilinearSteel, center: Union[Point, tuple[float, float]] = (0.0, 0.0)):
        self.size = size # size of the rebar (e.g., #4, #5, etc.)
        self.mat = bar_mat
        if isinstance(center, tuple):
            center = Point(*center)
        self._center = center
    
    @property
    def center(self) -> Point:
        return self._center
    
    @center.setter
    def center(self, new_center: Union[Point, tuple[float, float]]):
        if isinstance(new_center, tuple):
            new_center = Point(*new_center)
        self._center = new_center
              
    @property
    def shape(self) -> Circle:
        return Circle(radius = self.dia / 2, center = self._center)

    @property
    def fy(self) -> float:
        return self.mat.fy # ksi, yield strength
    
    @property
    def fu(self) -> float:
        return self.mat.fu # ksi, ultimate strength
    
    @property
    def area(self) -> float:
        return self.shape.area # in^2, cross-sectional area
    
    @property
    def dia(self) -> float:
        match self.size:
            case 3 | 4 | 5 | 6 | 7 | 8:
                return self.size / 8.0 # inch, diameter of the rebar section
            case 9:
                return 1.125
            case 10:
                return 1.270
            case 11:
                return 1.410
            case 14:
                return 1.693
            case 18:
                return 2.257
            case _:
                raise ValueError(f"Invalid rebar size: {self.size}. Valid sizes are 3-14 and 18.")
            
    def __repr__(self):
        return f"RebarSection(size={self.size}, dia={self.dia}, fy={self.fy}, fu={self.fu}, center={self._center}, )"
            
    def plot(self, ax:matplotlib.axes.Axes=None, **kwargs):
        """
        Plots the rebar section on the given axis.
        
        Args:
            ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure and axis will be created.
            **kwargs: Additional keyword arguments for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        self.shape.plot(ax=ax, **kwargs)
        
        return ax


def main():
    pass
    
    
if __name__ == "__main__":
    main()